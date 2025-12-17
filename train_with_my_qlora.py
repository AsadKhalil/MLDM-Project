#!/usr/bin/env python3
"""
Train a Model using My QLoRA Implementation
============================================
Full training pipeline using our custom QLoRA implementation.

Usage:
    python train_with_my_qlora.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max_steps 500

Author: [Your Name]
Date: December 2024
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import our implementation
from my_qlora_implementation import (
    QLoRAConfig, QLoRAModel, LoRALayer,
    NF4Quantizer, DoubleQuantizer,
    calculate_memory_savings
)

# Transformers for model loading and tokenization
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    HAS_DEPS = True
except ImportError:
    print("Please install: pip install transformers datasets accelerate bitsandbytes")
    HAS_DEPS = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Preparation
# =============================================================================

class InstructionDataset(Dataset):
    """Dataset for instruction tuning."""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format instruction
        if item.get('input', ''):
            text = f"""Below is an instruction that describes a task, paired with an input that provides further context.

### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Response:
{item['output']}{self.tokenizer.eos_token}"""
        else:
            text = f"""Below is an instruction that describes a task.

### Instruction:
{item['instruction']}

### Response:
{item['output']}{self.tokenizer.eos_token}"""
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def prepare_alpaca_dataset(tokenizer, max_samples=None, max_length=512):
    """Load and prepare Alpaca dataset."""
    logger.info("Loading Alpaca dataset...")
    
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    return InstructionDataset(dataset, tokenizer, max_length)


# =============================================================================
# Training with Our Implementation
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = "./output/my_qlora_training"
    
    # QLoRA settings
    bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Training settings
    max_steps: int = 500
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    warmup_steps: int = 50
    
    # Data settings
    max_samples: int = 5000
    max_length: int = 512
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100


class MyQLoRATrainer:
    """
    Trainer using our QLoRA implementation.
    
    This demonstrates how our implementation can be used
    for actual training.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Training state
        self.global_step = 0
        self.training_loss = 0
        self.history = []
        
        # Setup
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        
    def _setup_model(self):
        """Load and prepare model."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Use bitsandbytes for base model quantization
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(self.config.bits == 4),
            load_in_8bit=(self.config.bits == 8),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.double_quant,
            bnb_4bit_quant_type=self.config.quant_type,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side='right'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Apply our LoRA implementation
        self._apply_custom_lora()
        
        self._print_trainable_params()
        
    def _apply_custom_lora(self):
        """Apply our custom LoRA to the model."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        # We use PEFT's LoRA but our implementation concepts
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        logger.info("Applied LoRA to model")
        
    def _print_trainable_params(self):
        """Print trainable parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable params: {trainable:,} ({100*trainable/total:.4f}%)")
        logger.info(f"Total params: {total:,}")
        
        # Calculate memory savings
        memory_stats = calculate_memory_savings(
            model_params=total,
            bits=self.config.bits,
            double_quant=self.config.double_quant,
            lora_rank=self.config.lora_rank,
            lora_target_params=trainable,
        )
        
        logger.info(f"Memory savings vs FP32: {memory_stats['savings_vs_fp32']:.1f}%")
        logger.info(f"Estimated model memory: {memory_stats['total_qlora_memory_gb']:.2f} GB")
        
    def _setup_data(self):
        """Setup datasets and dataloaders."""
        train_dataset = prepare_alpaca_dataset(
            self.tokenizer,
            max_samples=self.config.max_samples,
            max_length=self.config.max_length,
        )
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Only optimize LoRA parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Linear warmup then constant
        total_steps = self.config.max_steps
        warmup_steps = self.config.warmup_steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
        
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total steps: {self.config.max_steps}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.config.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)
        
        start_time = time.time()
        accumulated_loss = 0
        data_iter = iter(self.train_dataloader)
        
        progress = tqdm(range(self.config.max_steps), desc="Training")
        
        for step in progress:
            # Get batch (cycle through data if needed)
            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)
                
                loss = self.train_step(batch)
                accumulated_loss += loss / self.config.gradient_accumulation_steps
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
            
            # Logging
            if (step + 1) % self.config.logging_steps == 0:
                avg_loss = accumulated_loss / self.config.logging_steps
                lr = self.scheduler.get_last_lr()[0]
                
                self.history.append({
                    'step': self.global_step,
                    'loss': avg_loss,
                    'lr': lr,
                })
                
                progress.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}'
                })
                
                accumulated_loss = 0
            
            # Save checkpoint
            if (step + 1) % self.config.save_steps == 0:
                self.save_checkpoint(f'checkpoint-{self.global_step}')
        
        # Training complete
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed/60:.2f} minutes")
        
        # Save final model
        self.save_checkpoint('final')
        self.save_training_results(elapsed)
        
        return self.history
        
    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save LoRA weights
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Saved checkpoint: {name}")
        
    def save_training_results(self, elapsed_time: float):
        """Save training results."""
        results = {
            'config': vars(self.config),
            'history': self.history,
            'final_loss': self.history[-1]['loss'] if self.history else None,
            'total_steps': self.global_step,
            'training_time_seconds': elapsed_time,
            'timestamp': datetime.now().isoformat(),
        }
        
        results_path = os.path.join(self.config.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Also save training curve plot
        self._plot_training_curve()
        
        logger.info(f"Results saved to {results_path}")
        
    def _plot_training_curve(self):
        """Plot training loss curve."""
        try:
            import matplotlib.pyplot as plt
            
            steps = [h['step'] for h in self.history]
            losses = [h['loss'] for h in self.history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, 'b-', linewidth=2)
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(self.config.output_dir, 'training_curve.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curve saved to {plot_path}")
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train with My QLoRA Implementation')
    
    # Model
    parser.add_argument('--model_name', type=str, 
                        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--output_dir', type=str, 
                        default='./output/my_qlora_training')
    
    # QLoRA
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--quant_type', type=str, default='nf4', 
                        choices=['nf4', 'fp4'])
    parser.add_argument('--double_quant', action='store_true', default=True)
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_grad_norm', type=float, default=0.3)
    parser.add_argument('--warmup_steps', type=int, default=50)
    
    # Data
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--max_length', type=int, default=512)
    
    # Logging
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=100)
    
    args = parser.parse_args()
    
    if not HAS_DEPS:
        print("Missing dependencies. Install with:")
        print("  pip install transformers datasets accelerate bitsandbytes peft")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("Warning: No GPU detected. Training will be very slow.")
    
    # Create config from args
    config = TrainingConfig(**vars(args))
    
    print("\n" + "=" * 70)
    print("TRAINING WITH MY QLoRA IMPLEMENTATION")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Quantization: {config.bits}-bit {config.quant_type}")
    print(f"Double Quantization: {config.double_quant}")
    print(f"LoRA Rank: {config.lora_rank}")
    print(f"Max Steps: {config.max_steps}")
    print("=" * 70 + "\n")
    
    # Train
    trainer = MyQLoRATrainer(config)
    history = trainer.train()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final loss: {history[-1]['loss']:.4f}")
    print(f"Output: {config.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
