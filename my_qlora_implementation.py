"""
My QLoRA Implementation from Scratch
====================================
Custom implementation of QLoRA: Efficient Finetuning of Quantized LLMs

This implements the core QLoRA techniques:
1. 4-bit NormalFloat (NF4) Quantization
2. Double Quantization
3. LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
4. Paged Optimizers for memory efficiency

Author: [Your Name]
Date: December 2024
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: NF4 Quantization Implementation
# =============================================================================

class NF4Quantizer:
    """
    NormalFloat 4-bit (NF4) Quantization
    
    NF4 is an information-theoretically optimal data type for normally 
    distributed weights. It creates 16 quantization bins that are optimally
    spaced for a normal distribution N(0, 1).
    
    Reference: QLoRA Paper Section 3.1
    """
    
    def __init__(self):
        # Pre-computed NF4 quantization levels for N(0,1)
        # These are the 16 optimal quantization points
        self.nf4_levels = torch.tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
        ])
        
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to NF4.
        
        Args:
            tensor: Input tensor to quantize
            
        Returns:
            quantized: 4-bit quantized indices
            scale: Per-block scaling factors
            zero_point: Zero points (all zeros for NF4)
        """
        # Compute per-tensor or per-block statistics
        absmax = tensor.abs().max()
        scale = absmax / self.nf4_levels.abs().max()
        
        # Normalize tensor
        normalized = tensor / (scale + 1e-8)
        
        # Find nearest NF4 level for each element
        nf4_levels = self.nf4_levels.to(tensor.device)
        distances = torch.abs(normalized.unsqueeze(-1) - nf4_levels)
        quantized = distances.argmin(dim=-1).to(torch.uint8)
        
        return quantized, scale, torch.tensor(0.0)
    
    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Dequantize NF4 tensor back to floating point.
        
        Args:
            quantized: 4-bit quantized indices
            scale: Scaling factor
            
        Returns:
            Dequantized floating point tensor
        """
        nf4_levels = self.nf4_levels.to(quantized.device)
        dequantized = nf4_levels[quantized.long()] * scale
        return dequantized


class FP4Quantizer:
    """
    Standard 4-bit Floating Point Quantization (for comparison)
    """
    
    def __init__(self, signed: bool = True):
        self.signed = signed
        self.num_bits = 4
        if signed:
            self.qmin = -8
            self.qmax = 7
        else:
            self.qmin = 0
            self.qmax = 15
            
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize to FP4."""
        vmin, vmax = tensor.min(), tensor.max()
        scale = (vmax - vmin) / (self.qmax - self.qmin)
        zero_point = self.qmin - vmin / (scale + 1e-8)
        
        quantized = torch.clamp(
            torch.round(tensor / (scale + 1e-8) + zero_point),
            self.qmin, self.qmax
        ).to(torch.uint8)
        
        return quantized, scale, zero_point
    
    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor, 
                   zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize FP4 tensor."""
        return (quantized.float() - zero_point) * scale


# =============================================================================
# PART 2: Double Quantization Implementation
# =============================================================================

class DoubleQuantizer:
    """
    Double Quantization for memory efficiency.
    
    Instead of storing full 32-bit scaling factors, we quantize the 
    scaling factors themselves to 8-bit, with a second-level scaling factor.
    
    Memory savings: ~0.37 bits per parameter
    
    Reference: QLoRA Paper Section 3.2
    """
    
    def __init__(self, block_size: int = 64):
        self.block_size = block_size
        self.nf4_quantizer = NF4Quantizer()
        
    def quantize_with_double_quant(
        self, 
        tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply double quantization.
        
        Args:
            tensor: Input weight tensor
            
        Returns:
            Dictionary containing all quantization parameters
        """
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        
        # Pad to block size
        num_elements = tensor_flat.numel()
        num_blocks = math.ceil(num_elements / self.block_size)
        padded_size = num_blocks * self.block_size
        
        if padded_size > num_elements:
            tensor_flat = F.pad(tensor_flat, (0, padded_size - num_elements))
        
        # Reshape into blocks
        tensor_blocks = tensor_flat.view(num_blocks, self.block_size)
        
        # First quantization: quantize each block
        block_absmax = tensor_blocks.abs().max(dim=1).values
        normalized_blocks = tensor_blocks / (block_absmax.unsqueeze(1) + 1e-8)
        
        # Quantize to NF4
        nf4_levels = self.nf4_quantizer.nf4_levels.to(tensor.device)
        distances = torch.abs(normalized_blocks.unsqueeze(-1) - nf4_levels)
        quantized_weights = distances.argmin(dim=-1).to(torch.uint8)
        
        # Second quantization: quantize the scaling factors
        scale_absmax = block_absmax.abs().max()
        normalized_scales = block_absmax / (scale_absmax + 1e-8)
        
        # Quantize scales to 8-bit
        quantized_scales = torch.clamp(
            torch.round(normalized_scales * 127),
            0, 255
        ).to(torch.uint8)
        
        return {
            'quantized_weights': quantized_weights,
            'quantized_scales': quantized_scales,
            'scale_scale': scale_absmax,
            'original_shape': original_shape,
            'num_elements': num_elements,
        }
    
    def dequantize_with_double_quant(
        self, 
        quant_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Dequantize double-quantized tensor.
        """
        # Recover scales
        scales = quant_dict['quantized_scales'].float() / 127 * quant_dict['scale_scale']
        
        # Recover weights
        nf4_levels = self.nf4_quantizer.nf4_levels.to(quant_dict['quantized_weights'].device)
        weights = nf4_levels[quant_dict['quantized_weights'].long()] * scales.unsqueeze(1)
        
        # Reshape and trim padding
        weights_flat = weights.flatten()[:quant_dict['num_elements']]
        return weights_flat.view(quant_dict['original_shape'])


# =============================================================================
# PART 3: LoRA Implementation
# =============================================================================

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) Layer
    
    LoRA decomposes weight updates into low-rank matrices:
    W' = W + BA where B ∈ R^(d×r) and A ∈ R^(r×k)
    
    This reduces trainable parameters from d×k to r×(d+k)
    
    Reference: LoRA Paper (Hu et al., 2021)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        alpha: float = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize LoRA matrices."""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros (so initial output is zero)
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation.
        
        Args:
            x: Input tensor
            base_output: Output from the base (frozen) layer
            
        Returns:
            Adapted output: base_output + scaling * (x @ A.T @ B.T)
        """
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_output + self.scaling * lora_output


class QuantizedLinearWithLoRA(nn.Module):
    """
    Quantized Linear Layer with LoRA adaptation.
    
    Combines:
    1. NF4/FP4 quantized base weights (frozen)
    2. LoRA low-rank adaptation (trainable)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_type: str = 'nf4',
        double_quant: bool = True,
        lora_rank: int = 64,
        lora_alpha: float = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        self.double_quant = double_quant
        
        # Initialize quantizer
        if quant_type == 'nf4':
            self.quantizer = NF4Quantizer()
        else:
            self.quantizer = FP4Quantizer()
            
        if double_quant:
            self.double_quantizer = DoubleQuantizer()
        
        # Quantized weight storage (will be set by quantize_weights)
        self.register_buffer('quant_weight', None)
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_zero_point', None)
        
        # For double quantization
        self.quant_dict = None
        
        # Bias (not quantized)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        # LoRA layers
        self.lora = LoRALayer(
            in_features, out_features, 
            lora_rank, lora_alpha, lora_dropout
        )
        
    def quantize_weights(self, weight: torch.Tensor):
        """Quantize and store weights."""
        if self.double_quant:
            self.quant_dict = self.double_quantizer.quantize_with_double_quant(weight)
        else:
            quant, scale, zp = self.quantizer.quantize(weight)
            self.quant_weight = quant
            self.weight_scale = scale
            self.weight_zero_point = zp
            
    def get_dequantized_weight(self) -> torch.Tensor:
        """Get dequantized weights for forward pass."""
        if self.double_quant and self.quant_dict is not None:
            return self.double_quantizer.dequantize_with_double_quant(self.quant_dict)
        elif self.quant_weight is not None:
            return self.quantizer.dequantize(
                self.quant_weight, self.weight_scale, self.weight_zero_point
            )
        else:
            raise ValueError("Weights not quantized yet!")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights + LoRA."""
        # Get dequantized base weights
        weight = self.get_dequantized_weight()
        
        # Base linear transformation
        base_output = F.linear(x, weight, self.bias)
        
        # Add LoRA adaptation
        output = self.lora(x, base_output)
        
        return output


# =============================================================================
# PART 4: QLoRA Model Wrapper
# =============================================================================

@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning."""
    # Quantization
    bits: int = 4
    quant_type: str = 'nf4'  # 'nf4' or 'fp4'
    double_quant: bool = True
    
    # LoRA
    lora_rank: int = 64
    lora_alpha: float = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'
    ])
    
    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03


class QLoRAModel(nn.Module):
    """
    QLoRA Model Wrapper
    
    Wraps a base model with:
    1. Quantized frozen weights
    2. LoRA adapters for fine-tuning
    """
    
    def __init__(self, base_model: nn.Module, config: QLoRAConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Apply quantization and LoRA
        self._prepare_model()
        
    def _prepare_model(self):
        """Prepare model: freeze base, add LoRA to target modules."""
        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Find and replace target modules with LoRA versions
        self._replace_modules_with_lora(self.base_model)
        
    def _replace_modules_with_lora(self, module: nn.Module, prefix: str = ''):
        """Recursively replace linear layers with LoRA versions."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                # Check if this is a target module
                is_target = any(
                    target in full_name 
                    for target in self.config.lora_target_modules
                )
                
                if is_target:
                    # Replace with quantized + LoRA version
                    new_layer = QuantizedLinearWithLoRA(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        quant_type=self.config.quant_type,
                        double_quant=self.config.double_quant,
                        lora_rank=self.config.lora_rank,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                    )
                    
                    # Quantize original weights
                    new_layer.quantize_weights(child.weight.data)
                    if child.bias is not None:
                        new_layer.bias.data = child.bias.data.clone()
                        
                    setattr(module, name, new_layer)
                    logger.info(f"Applied QLoRA to: {full_name}")
            else:
                self._replace_modules_with_lora(child, full_name)
                
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get only LoRA parameters for training."""
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params
    
    def print_trainable_parameters(self):
        """Print trainable vs total parameters."""
        trainable = 0
        total = 0
        
        for name, param in self.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
                
        trainable_percent = 100 * trainable / total
        print(f"Trainable params: {trainable:,} || "
              f"Total params: {total:,} || "
              f"Trainable%: {trainable_percent:.4f}%")
        
        return trainable, total
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights."""
        lora_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                lora_state_dict[name] = param.data.cpu()
                
        torch.save({
            'lora_state_dict': lora_state_dict,
            'config': self.config,
        }, path)
        logger.info(f"Saved LoRA weights to {path}")
        
    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        checkpoint = torch.load(path)
        lora_state_dict = checkpoint['lora_state_dict']
        
        for name, param in self.named_parameters():
            if name in lora_state_dict:
                param.data = lora_state_dict[name].to(param.device)
                
        logger.info(f"Loaded LoRA weights from {path}")


# =============================================================================
# PART 5: Training Utilities
# =============================================================================

class QLoRATrainer:
    """
    Simple trainer for QLoRA fine-tuning.
    """
    
    def __init__(
        self,
        model: QLoRAModel,
        config: QLoRAConfig,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer with only LoRA parameters
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.global_step = 0
        self.training_history = []
        
    def train_step(
        self, 
        input_ids: torch.Tensor, 
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Single training step."""
        self.model.train()
        
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_parameters(),
            self.config.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
        self.training_history.append({
            'step': self.global_step,
            'loss': loss.item(),
        })
        
        return loss.item()
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        self.model.save_lora_weights(os.path.join(path, 'lora_weights.pt'))
        
        with open(os.path.join(path, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
            
        logger.info(f"Checkpoint saved to {path}")


# =============================================================================
# PART 6: Memory Calculation Utilities
# =============================================================================

def calculate_memory_savings(
    model_params: int,
    bits: int = 4,
    double_quant: bool = True,
    lora_rank: int = 64,
    lora_target_params: int = None,
) -> Dict[str, float]:
    """
    Calculate memory savings from QLoRA.
    
    Args:
        model_params: Total model parameters
        bits: Quantization bits (4 or 8)
        double_quant: Whether double quantization is used
        lora_rank: LoRA rank
        lora_target_params: Parameters in LoRA target modules
        
    Returns:
        Dictionary with memory calculations
    """
    # Full precision (FP32)
    fp32_memory = model_params * 4  # 4 bytes per param
    
    # FP16
    fp16_memory = model_params * 2
    
    # Quantized
    quant_memory = model_params * bits / 8
    
    # Double quantization saves ~0.37 bits per param
    if double_quant:
        quant_memory -= model_params * 0.37 / 8
        
    # LoRA parameters (FP16)
    if lora_target_params:
        # LoRA adds 2 * rank * (in + out) parameters per target layer
        lora_params = 2 * lora_rank * lora_target_params
        lora_memory = lora_params * 2  # FP16
    else:
        lora_memory = 0
        
    total_qlora_memory = quant_memory + lora_memory
    
    return {
        'fp32_memory_gb': fp32_memory / 1e9,
        'fp16_memory_gb': fp16_memory / 1e9,
        'quantized_memory_gb': quant_memory / 1e9,
        'lora_memory_gb': lora_memory / 1e9,
        'total_qlora_memory_gb': total_qlora_memory / 1e9,
        'savings_vs_fp32': (1 - total_qlora_memory / fp32_memory) * 100,
        'savings_vs_fp16': (1 - total_qlora_memory / fp16_memory) * 100,
    }


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_quantization():
    """Demonstrate quantization implementations."""
    print("=" * 60)
    print("QLoRA Implementation Demo")
    print("=" * 60)
    
    # Create sample tensor (simulating model weights)
    torch.manual_seed(42)
    sample_weights = torch.randn(1024, 1024)  # 1M parameters
    
    print(f"\nOriginal tensor shape: {sample_weights.shape}")
    print(f"Original memory: {sample_weights.numel() * 4 / 1e6:.2f} MB (FP32)")
    
    # Test NF4 Quantization
    print("\n--- NF4 Quantization ---")
    nf4_quantizer = NF4Quantizer()
    quant, scale, zp = nf4_quantizer.quantize(sample_weights)
    dequant = nf4_quantizer.dequantize(quant, scale)
    
    nf4_memory = quant.numel() * 0.5 / 1e6  # 4 bits = 0.5 bytes
    nf4_error = (sample_weights - dequant).abs().mean().item()
    print(f"Quantized memory: {nf4_memory:.2f} MB")
    print(f"Mean absolute error: {nf4_error:.6f}")
    
    # Test FP4 Quantization
    print("\n--- FP4 Quantization ---")
    fp4_quantizer = FP4Quantizer()
    quant_fp4, scale_fp4, zp_fp4 = fp4_quantizer.quantize(sample_weights)
    dequant_fp4 = fp4_quantizer.dequantize(quant_fp4, scale_fp4, zp_fp4)
    
    fp4_error = (sample_weights - dequant_fp4).abs().mean().item()
    print(f"Mean absolute error: {fp4_error:.6f}")
    
    # Test Double Quantization
    print("\n--- Double Quantization ---")
    double_quantizer = DoubleQuantizer(block_size=64)
    quant_dict = double_quantizer.quantize_with_double_quant(sample_weights)
    dequant_double = double_quantizer.dequantize_with_double_quant(quant_dict)
    
    double_error = (sample_weights - dequant_double).abs().mean().item()
    print(f"Mean absolute error: {double_error:.6f}")
    
    # Memory calculation
    print("\n--- Memory Savings (7B Model Example) ---")
    memory_stats = calculate_memory_savings(
        model_params=7_000_000_000,
        bits=4,
        double_quant=True,
        lora_rank=64,
        lora_target_params=100_000_000,  # ~100M params in target modules
    )
    
    for key, value in memory_stats.items():
        print(f"{key}: {value:.2f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_quantization()
