#!/usr/bin/env python3
"""
Run Original QLoRA Implementation and Save Results
===================================================
This script runs the original QLoRA implementation from the paper
and saves all results for analysis.

Usage:
    # Quick test (100 steps)
    python run_original_qlora.py --quick_test

    # Baseline experiment
    python run_original_qlora.py --experiment baseline

    # Compare NF4 vs FP4
    python run_original_qlora.py --experiment fp4_comparison

    # Full experiment suite
    python run_original_qlora.py --run_all
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Add qlora directory to path
QLORA_DIR = Path(__file__).parent / "qlora"
sys.path.insert(0, str(QLORA_DIR))


# =============================================================================
# Experiment Configurations
# =============================================================================

# Model options (choose based on your GPU)
MODELS = {
    'tiny': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # ~3GB VRAM
    'small': 'microsoft/phi-2',                      # ~6GB VRAM
    '7b': 'huggyllama/llama-7b',                    # ~6GB VRAM with 4-bit
}

EXPERIMENTS = {
    'quick_test': {
        'description': 'Quick test run (100 steps)',
        'max_steps': 100,
        'bits': 4,
        'quant_type': 'nf4',
        'double_quant': True,
        'lora_r': 64,
        'lora_alpha': 16,
        'save_steps': 50,
    },
    'baseline': {
        'description': 'Baseline: 4-bit NF4 + Double Quant',
        'max_steps': 500,
        'bits': 4,
        'quant_type': 'nf4',
        'double_quant': True,
        'lora_r': 64,
        'lora_alpha': 16,
        'save_steps': 100,
    },
    'fp4_comparison': {
        'description': 'Comparison: 4-bit FP4 + Double Quant',
        'max_steps': 500,
        'bits': 4,
        'quant_type': 'fp4',
        'double_quant': True,
        'lora_r': 64,
        'lora_alpha': 16,
        'save_steps': 100,
    },
    'no_double_quant': {
        'description': 'Ablation: NF4 without Double Quant',
        'max_steps': 500,
        'bits': 4,
        'quant_type': 'nf4',
        'double_quant': False,
        'lora_r': 64,
        'lora_alpha': 16,
        'save_steps': 100,
    },
    'lora_r16': {
        'description': 'Ablation: Lower LoRA rank (r=16)',
        'max_steps': 500,
        'bits': 4,
        'quant_type': 'nf4',
        'double_quant': True,
        'lora_r': 16,
        'lora_alpha': 16,
        'save_steps': 100,
    },
    'lora_r128': {
        'description': 'Ablation: Higher LoRA rank (r=128)',
        'max_steps': 500,
        'bits': 4,
        'quant_type': 'nf4',
        'double_quant': True,
        'lora_r': 128,
        'lora_alpha': 16,
        'save_steps': 100,
    },
}


# =============================================================================
# Run Experiment Functions
# =============================================================================

def check_cuda_compatibility():
    """Check if CUDA is compatible with current PyTorch."""
    import torch
    if torch.cuda.is_available():
        try:
            # Try to create a tensor on GPU
            _ = torch.zeros(1).cuda()
            return True
        except Exception as e:
            print(f"CUDA compatibility issue: {e}")
            return False
    return False


def run_original_qlora(
    model_name: str,
    output_dir: str,
    experiment_config: dict,
    dataset: str = 'alpaca',
    max_samples: int = None,
    use_bf16: bool = True,
    batch_size: int = 4,
):
    """
    Run the original qlora.py script with given configuration.
    Uses qlora_fixed.py which is compatible with newer transformers.
    """
    
    # Check CUDA compatibility
    cuda_works = check_cuda_compatibility()
    if not cuda_works:
        print("\n⚠️  CUDA not compatible! Using smaller batch size and fp16=False")
        use_bf16 = False
        batch_size = 1
    
    # Use the FIXED version of qlora.py
    qlora_script = QLORA_DIR / 'qlora_fixed.py'
    if not qlora_script.exists():
        print(f"Warning: qlora_fixed.py not found, using original qlora.py")
        qlora_script = QLORA_DIR / 'qlora.py'
    
    # Build command
    cmd = [
        'python', str(qlora_script),
        '--model_name_or_path', model_name,
        '--output_dir', output_dir,
        '--dataset', dataset,
        '--do_train',
        '--do_eval',
        
        # Quantization settings
        '--bits', str(experiment_config['bits']),
        '--quant_type', experiment_config['quant_type'],
        
        # LoRA settings
        '--lora_r', str(experiment_config['lora_r']),
        '--lora_alpha', str(experiment_config['lora_alpha']),
        '--lora_dropout', '0.1',
        
        # Training settings
        '--max_steps', str(experiment_config['max_steps']),
        '--per_device_train_batch_size', str(batch_size),
        '--gradient_accumulation_steps', '4',
        '--learning_rate', '0.0002',
        '--warmup_ratio', '0.03',
        '--lr_scheduler_type', 'constant',
        '--gradient_checkpointing',
        '--optim', 'paged_adamw_32bit',
        
        # Logging
        '--logging_steps', '10',
        '--save_steps', str(experiment_config['save_steps']),
        '--save_total_limit', '3',
        '--report_to', 'none',
    ]
    
    # Add double quantization flag
    if experiment_config['double_quant']:
        cmd.append('--double_quant')
    
    # Add bf16 if supported and CUDA works
    if use_bf16 and cuda_works:
        cmd.append('--bf16')
    else:
        cmd.append('--fp16')
    
    # Limit samples for faster testing
    if max_samples:
        cmd.extend(['--max_train_samples', str(max_samples)])
    
    return cmd


def run_experiment(
    experiment_name: str,
    model_size: str = 'tiny',
    output_base: str = './output',
    max_samples: int = None,
):
    """
    Run a single experiment and save results.
    """
    
    if experiment_name not in EXPERIMENTS:
        print(f"Unknown experiment: {experiment_name}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        return None
    
    config = EXPERIMENTS[experiment_name]
    model_name = MODELS[model_size]
    output_dir = os.path.join(output_base, f'original_{experiment_name}')
    
    print("\n" + "=" * 70)
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    print("=" * 70)
    print(f"Description: {config['description']}")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Max Steps: {config['max_steps']}")
    print("=" * 70 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save experiment config
    experiment_info = {
        'experiment_name': experiment_name,
        'config': config,
        'model_name': model_name,
        'model_size': model_size,
        'output_dir': output_dir,
        'max_samples': max_samples,
        'start_time': datetime.now().isoformat(),
    }
    
    with open(os.path.join(output_dir, 'experiment_config.json'), 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    # Build and run command
    cmd = run_original_qlora(
        model_name=model_name,
        output_dir=output_dir,
        experiment_config=config,
        max_samples=max_samples,
    )
    
    print("Running command:")
    print(" ".join(cmd))
    print("\n")
    
    start_time = time.time()
    
    try:
        # Run the training
        result = subprocess.run(
            cmd,
            cwd=str(QLORA_DIR),
            capture_output=False,  # Show output in real-time
            text=True,
        )
        
        elapsed_time = time.time() - start_time
        success = result.returncode == 0
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        success = False
        print(f"Error: {e}")
    
    # Save results summary
    results = {
        'experiment_name': experiment_name,
        'success': success,
        'elapsed_time_seconds': elapsed_time,
        'elapsed_time_minutes': elapsed_time / 60,
        'end_time': datetime.now().isoformat(),
    }
    
    # Try to load training metrics
    trainer_state_path = os.path.join(output_dir, 'trainer_state.json')
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)
        
        # Extract final metrics
        log_history = trainer_state.get('log_history', [])
        
        train_losses = [l['loss'] for l in log_history if 'loss' in l]
        eval_losses = [l['eval_loss'] for l in log_history if 'eval_loss' in l]
        
        results['final_train_loss'] = train_losses[-1] if train_losses else None
        results['final_eval_loss'] = eval_losses[-1] if eval_losses else None
        results['total_steps'] = trainer_state.get('global_step', 0)
        results['training_history'] = log_history
    
    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Success: {success}")
    print(f"Time: {elapsed_time/60:.2f} minutes")
    if results.get('final_train_loss'):
        print(f"Final Train Loss: {results['final_train_loss']:.4f}")
    if results.get('final_eval_loss'):
        print(f"Final Eval Loss: {results['final_eval_loss']:.4f}")
    print(f"Results saved: {results_path}")
    print("=" * 70 + "\n")
    
    return results


def run_all_experiments(
    model_size: str = 'tiny',
    output_base: str = './output',
    max_samples: int = 2000,
    experiments: list = None,
):
    """
    Run multiple experiments and compile results.
    """
    
    experiments = experiments or ['baseline', 'fp4_comparison', 'no_double_quant', 'lora_r16']
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'model_size': model_size,
        'experiments': {},
    }
    
    for exp_name in experiments:
        print(f"\n{'#' * 70}")
        print(f"# Starting: {exp_name}")
        print(f"{'#' * 70}\n")
        
        results = run_experiment(
            experiment_name=exp_name,
            model_size=model_size,
            output_base=output_base,
            max_samples=max_samples,
        )
        
        if results:
            all_results['experiments'][exp_name] = results
    
    # Save combined results
    combined_path = os.path.join(output_base, 'all_experiments_results.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate comparison table
    generate_comparison_table(all_results, output_base)
    
    return all_results


def generate_comparison_table(results: dict, output_dir: str):
    """
    Generate a comparison table of all experiments.
    """
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)
    
    header = f"{'Experiment':<20} {'Train Loss':>12} {'Eval Loss':>12} {'Time (min)':>12}"
    print(header)
    print("-" * 70)
    
    rows = []
    for exp_name, exp_results in results.get('experiments', {}).items():
        train_loss = exp_results.get('final_train_loss', 'N/A')
        eval_loss = exp_results.get('final_eval_loss', 'N/A')
        time_min = exp_results.get('elapsed_time_minutes', 0)
        
        if isinstance(train_loss, float):
            train_loss = f"{train_loss:.4f}"
        if isinstance(eval_loss, float):
            eval_loss = f"{eval_loss:.4f}"
        
        row = f"{exp_name:<20} {train_loss:>12} {eval_loss:>12} {time_min:>12.2f}"
        print(row)
        rows.append({
            'experiment': exp_name,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'time_min': time_min,
        })
    
    print("=" * 70)
    
    # Save as markdown
    md_content = f"""# Original QLoRA Experiment Results

## Date: {results.get('timestamp', 'N/A')}
## Model: {results.get('model_size', 'N/A')}

## Results Comparison

| Experiment | Train Loss | Eval Loss | Time (min) |
|------------|------------|-----------|------------|
"""
    
    for row in rows:
        md_content += f"| {row['experiment']} | {row['train_loss']} | {row['eval_loss']} | {row['time_min']:.2f} |\n"
    
    md_content += """
## Key Findings

1. **NF4 vs FP4**: Compare baseline vs fp4_comparison
2. **Double Quantization**: Compare baseline vs no_double_quant  
3. **LoRA Rank**: Compare baseline vs lora_r16

## Files Generated

Each experiment creates:
- `experiment_config.json` - Configuration used
- `results.json` - Final metrics
- `trainer_state.json` - Training history
- `checkpoint-*/` - Model checkpoints
"""
    
    md_path = os.path.join(output_dir, 'EXPERIMENT_RESULTS.md')
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    print(f"\nResults saved to: {md_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run Original QLoRA and Save Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (100 steps, ~5-10 minutes)
    python run_original_qlora.py --quick_test

    # Run baseline experiment
    python run_original_qlora.py --experiment baseline

    # Compare NF4 vs FP4
    python run_original_qlora.py --experiment fp4_comparison

    # Run all experiments
    python run_original_qlora.py --run_all

    # Use a specific model size
    python run_original_qlora.py --experiment baseline --model_size 7b
        """
    )
    
    parser.add_argument('--experiment', type=str, 
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiment to run')
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick test (100 steps)')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all experiments')
    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=list(MODELS.keys()),
                        help='Model size to use')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='Max training samples (for faster runs)')
    
    args = parser.parse_args()
    
    # Check qlora.py exists
    if not (QLORA_DIR / 'qlora.py').exists():
        print(f"Error: qlora.py not found in {QLORA_DIR}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("ORIGINAL QLoRA EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Model: {MODELS[args.model_size]}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    if args.quick_test:
        run_experiment(
            experiment_name='quick_test',
            model_size=args.model_size,
            output_base=args.output_dir,
            max_samples=500,  # Very limited for quick test
        )
    elif args.run_all:
        run_all_experiments(
            model_size=args.model_size,
            output_base=args.output_dir,
            max_samples=args.max_samples,
        )
    elif args.experiment:
        run_experiment(
            experiment_name=args.experiment,
            model_size=args.model_size,
            output_base=args.output_dir,
            max_samples=args.max_samples,
        )
    else:
        parser.print_help()
        print("\n⚠️  Please specify --quick_test, --experiment, or --run_all")


if __name__ == "__main__":
    main()
