#!/usr/bin/env python3
"""
QLoRA Experiment Results Analyzer
Extracts and summarizes training metrics from QLoRA experiments
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

sns.set_style("whitegrid")


def load_trainer_state(output_dir: Path) -> Optional[Dict]:
    """Load trainer_state.json from an experiment directory."""
    state_file = output_dir / "trainer_state.json"
    if not state_file.exists():
        print(f"Warning: No trainer_state.json found in {output_dir}")
        return None
    
    with open(state_file) as f:
        return json.load(f)


def extract_metrics(state: Dict, exp_name: str) -> Dict:
    """Extract key metrics from trainer state."""
    if not state:
        return {}
    
    log_history = state.get('log_history', [])
    
    # Get final training loss
    train_losses = [log['loss'] for log in log_history if 'loss' in log]
    final_train_loss = train_losses[-1] if train_losses else None
    
    # Get final eval loss
    eval_losses = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
    final_eval_loss = eval_losses[-1] if eval_losses else None
    
    # Get eval accuracy if available
    eval_acc = None
    for log in reversed(log_history):
        if 'eval_accuracy' in log:
            eval_acc = log['eval_accuracy']
            break
    
    # Get training time info
    total_flos = state.get('total_flos', 0)
    
    return {
        'experiment': exp_name,
        'global_step': state.get('global_step', 0),
        'final_train_loss': final_train_loss,
        'final_eval_loss': final_eval_loss,
        'eval_accuracy': eval_acc,
        'total_flos': total_flos,
        'num_train_samples': len(train_losses),
        'num_eval_samples': len(eval_losses)
    }


def load_mmlu_results(output_dir: Path) -> Optional[Dict]:
    """Load MMLU evaluation results if available."""
    mmlu_file = output_dir / "mmlu_results.json"
    if mmlu_file.exists():
        with open(mmlu_file) as f:
            return json.load(f)
    return None


def analyze_experiment(output_dir: Path) -> Dict:
    """Analyze a single experiment directory."""
    exp_name = output_dir.name
    
    # Load trainer state
    state = load_trainer_state(output_dir)
    metrics = extract_metrics(state, exp_name)
    
    # Load MMLU results
    mmlu = load_mmlu_results(output_dir)
    if mmlu:
        metrics['mmlu_accuracy'] = mmlu.get('accuracy', None)
        metrics['mmlu_categories'] = mmlu.get('categories', {})
    
    # Check checkpoint sizes
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if checkpoints:
        total_size = sum(
            sum(f.stat().st_size for f in cp.rglob('*') if f.is_file())
            for cp in checkpoints
        )
        metrics['checkpoint_size_gb'] = total_size / (1024**3)
        metrics['num_checkpoints'] = len(checkpoints)
    
    return metrics


def analyze_all_experiments(base_dir: Path, pattern: str = "exp_*") -> pd.DataFrame:
    """Analyze all experiments matching the pattern."""
    results = []
    
    exp_dirs = sorted(base_dir.glob(pattern))
    
    if not exp_dirs:
        print(f"No experiments found matching pattern '{pattern}' in {base_dir}")
        return pd.DataFrame()
    
    for exp_dir in exp_dirs:
        if exp_dir.is_dir():
            print(f"Analyzing {exp_dir.name}...")
            metrics = analyze_experiment(exp_dir)
            if metrics:
                results.append(metrics)
    
    return pd.DataFrame(results)


def plot_loss_curves(output_dir: Path, save_path: Optional[Path] = None):
    """Plot training and evaluation loss curves."""
    state = load_trainer_state(output_dir)
    if not state:
        return
    
    log_history = state.get('log_history', [])
    
    # Extract losses
    train_data = [(log['step'], log['loss']) 
                  for log in log_history if 'loss' in log and 'step' in log]
    eval_data = [(log['step'], log['eval_loss']) 
                 for log in log_history if 'eval_loss' in log and 'step' in log]
    
    if not train_data:
        print("No training data found for plotting")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if train_data:
        steps, losses = zip(*train_data)
        ax.plot(steps, losses, label='Training Loss', linewidth=2)
    
    if eval_data:
        steps, losses = zip(*eval_data)
        ax.plot(steps, losses, label='Evaluation Loss', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Loss Curves: {output_dir.name}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(df: pd.DataFrame, metric: str, save_path: Optional[Path] = None):
    """Create comparison plot for a specific metric across experiments."""
    if df.empty or metric not in df.columns:
        print(f"Cannot plot {metric}: no data available")
        return
    
    # Remove rows with None values for this metric
    df_clean = df[df[metric].notna()].copy()
    
    if df_clean.empty:
        print(f"No valid data for {metric}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot
    x_pos = range(len(df_clean))
    ax.bar(x_pos, df_clean[metric], alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_clean['experiment'], rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Comparison: {metric.replace("_", " ").title()}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(df_clean[metric]):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_summary_report(df: pd.DataFrame, output_file: Path):
    """Generate a summary report in markdown format."""
    with open(output_file, 'w') as f:
        f.write("# QLoRA Experiments Summary Report\n\n")
        f.write(f"**Total Experiments**: {len(df)}\n\n")
        
        f.write("## Overall Statistics\n\n")
        f.write("### Key Metrics Summary\n\n")
        
        # Summary statistics table
        metrics_to_summarize = [
            'final_train_loss', 'final_eval_loss', 
            'eval_accuracy', 'mmlu_accuracy'
        ]
        
        for metric in metrics_to_summarize:
            if metric in df.columns and df[metric].notna().any():
                f.write(f"\n**{metric.replace('_', ' ').title()}:**\n")
                f.write(f"- Mean: {df[metric].mean():.4f}\n")
                f.write(f"- Std: {df[metric].std():.4f}\n")
                f.write(f"- Min: {df[metric].min():.4f}\n")
                f.write(f"- Max: {df[metric].max():.4f}\n")
        
        f.write("\n## Detailed Results\n\n")
        f.write(df.to_markdown(index=False))
        
        f.write("\n\n## Best Configurations\n\n")
        
        # Best by different metrics
        if 'final_eval_loss' in df.columns and df['final_eval_loss'].notna().any():
            best_loss = df.loc[df['final_eval_loss'].idxmin()]
            f.write(f"**Best Eval Loss**: {best_loss['experiment']} "
                   f"(loss: {best_loss['final_eval_loss']:.4f})\n\n")
        
        if 'mmlu_accuracy' in df.columns and df['mmlu_accuracy'].notna().any():
            best_mmlu = df.loc[df['mmlu_accuracy'].idxmax()]
            f.write(f"**Best MMLU Accuracy**: {best_mmlu['experiment']} "
                   f"(accuracy: {best_mmlu['mmlu_accuracy']:.2f}%)\n\n")
        
        if 'checkpoint_size_gb' in df.columns and df['checkpoint_size_gb'].notna().any():
            smallest = df.loc[df['checkpoint_size_gb'].idxmin()]
            f.write(f"**Smallest Checkpoints**: {smallest['experiment']} "
                   f"(size: {smallest['checkpoint_size_gb']:.2f} GB)\n\n")
    
    print(f"Summary report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze QLoRA experiment results"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='qlora/output',
        help='Base directory containing experiment outputs'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='exp_*',
        help='Pattern to match experiment directories'
    )
    parser.add_argument(
        '--plot_individual',
        action='store_true',
        help='Plot loss curves for each experiment'
    )
    parser.add_argument(
        '--compare_metric',
        type=str,
        choices=['final_train_loss', 'final_eval_loss', 'mmlu_accuracy'],
        help='Create comparison plot for specified metric'
    )
    parser.add_argument(
        '--save_plots',
        type=str,
        help='Directory to save plots'
    )
    parser.add_argument(
        '--report',
        type=str,
        default='experiment_summary.md',
        help='Output file for summary report'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    base_dir = Path(args.output_dir)
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Analyze all experiments
    print(f"\nAnalyzing experiments in {base_dir}...")
    df = analyze_all_experiments(base_dir, args.pattern)
    
    if df.empty:
        print("No experiments found to analyze")
        return
    
    print(f"\nFound {len(df)} experiments")
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save detailed CSV
    csv_path = Path(args.report).with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to {csv_path}")
    
    # Generate markdown report
    generate_summary_report(df, Path(args.report))
    
    # Plot individual experiments
    if args.plot_individual:
        plot_dir = Path(args.save_plots) if args.save_plots else None
        if plot_dir:
            plot_dir.mkdir(exist_ok=True, parents=True)
        
        for exp_dir in base_dir.glob(args.pattern):
            if exp_dir.is_dir():
                save_path = plot_dir / f"{exp_dir.name}_loss.png" if plot_dir else None
                plot_loss_curves(exp_dir, save_path)
    
    # Create comparison plot
    if args.compare_metric:
        save_path = None
        if args.save_plots:
            plot_dir = Path(args.save_plots)
            plot_dir.mkdir(exist_ok=True, parents=True)
            save_path = plot_dir / f"comparison_{args.compare_metric}.png"
        
        plot_comparison(df, args.compare_metric, save_path)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

