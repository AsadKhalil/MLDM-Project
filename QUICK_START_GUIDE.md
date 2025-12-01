# QLoRA Quick Start Guide

## 🚀 Getting Started in 15 Minutes

This guide will help you run your first QLoRA experiment quickly.

---

## Step 1: Environment Setup (5 minutes)

### Check GPU Availability
```bash
# Check if GPU is available
nvidia-smi

# Check CUDA version
nvcc --version
```

### Install Dependencies
```bash
cd /home/tk-lpt-0806/Desktop/MSDS/SEM3/Machine\ Learning\ and\ Data\ Mining/MLDM-Project/qlora

# Install requirements
pip install -U -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import peft; print(f'peft: {peft.__version__}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA Available: True
bitsandbytes: 0.40.0
transformers: 4.31.0
peft: 0.4.0
```

---

## Step 2: Setup Weights & Biases (Optional but Recommended) (3 minutes)

```bash
# Install wandb if not already installed
pip install wandb

# Login to wandb
wandb login

# Enter your API key when prompted
# Get it from: https://wandb.ai/authorize
```

If you don't want to use wandb, use `--report_to none` in commands below.

---

## Step 3: Run Your First Experiment (7 minutes setup + training time)

### Quick Test Run (Fast - ~10 minutes on single GPU)

```bash
# Navigate to qlora directory
cd qlora

# Run a quick test with minimal steps
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/quick_test \
    --dataset alpaca \
    --do_train True \
    --do_eval True \
    --bits 4 \
    --quant_type nf4 \
    --double_quant \
    --lora_r 64 \
    --lora_alpha 16 \
    --learning_rate 0.0002 \
    --max_steps 100 \
    --logging_steps 10 \
    --save_steps 50 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --bf16 \
    --report_to wandb \
    --run_name quick_test
```

**What this does:**
- Downloads LLaMA-7B model (happens once, ~13GB)
- Downloads Alpaca dataset (happens once)
- Trains for 100 steps (~10 minutes)
- Saves checkpoint at step 50 and 100

### Full Baseline Experiment (4-6 hours)

Once the quick test works, run a full experiment:

```bash
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/baseline_experiment \
    --dataset alpaca \
    --do_train True \
    --do_eval True \
    --do_mmlu_eval True \
    --bits 4 \
    --quant_type nf4 \
    --double_quant \
    --lora_r 64 \
    --lora_alpha 16 \
    --learning_rate 0.0002 \
    --max_steps 2000 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --bf16 \
    --report_to wandb \
    --run_name baseline_full
```

---

## Step 4: Monitor Your Training

### Watch GPU Usage
Open a new terminal:
```bash
watch -n 1 nvidia-smi
```

### Monitor Training Progress
```bash
# Watch the output directory
ls -lh output/baseline_experiment/

# Check logs (if using wandb)
# Visit: https://wandb.ai/your-username/qlora
```

### Expected Output
You should see:
```
loading base model huggyllama/llama-7b...
trainable params: 67108864 || all params: 6738415616 || trainable%: 0.9959
Starting training...
{'loss': 2.xxx, 'learning_rate': 0.0002, 'epoch': 0.xx}
...
```

---

## Step 5: Check Results

### Training Metrics
```bash
# View training logs
cat output/baseline_experiment/trainer_state.json | jq '.log_history'

# Check checkpoints
ls -lh output/baseline_experiment/checkpoint-*
```

### GPU Memory Usage
Typical memory usage for LLaMA-7B:
- 4-bit quantization: ~6-8 GB
- 8-bit quantization: ~10-14 GB
- Without quantization: ~28+ GB

### Evaluate Checkpoint
```bash
# Load and test generation
python examples/guanaco_generate.py \
    --model_name_or_path ./output/baseline_experiment/checkpoint-2000 \
    --base_model_name_or_path huggyllama/llama-7b
```

---

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory
**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size
--per_device_train_batch_size 1

# Increase gradient accumulation
--gradient_accumulation_steps 32

# Enable gradient checkpointing (should already be on)
--gradient_checkpointing

# Reduce max memory
--max_memory_MB 40000
```

### Issue 2: Model Download Fails
**Error:** Connection timeout or download error

**Solutions:**
```bash
# Use HuggingFace cache
export HF_HOME=/path/to/large/storage/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# Manually download model first
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('huggyllama/llama-7b')"
```

### Issue 3: Dataset Not Found
**Error:** `Dataset 'alpaca' not found`

**Solutions:**
```bash
# The dataset is downloaded automatically from HuggingFace
# If it fails, check internet connection

# Or use a local dataset
python qlora.py --dataset /path/to/local/dataset.json
```

### Issue 4: bitsandbytes Import Error
**Error:** `ImportError: cannot import name 'BitsAndBytesConfig'`

**Solutions:**
```bash
# Reinstall bitsandbytes
pip uninstall bitsandbytes
pip install bitsandbytes==0.40.0

# Check CUDA compatibility
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

### Issue 5: Too Slow Training
**Symptoms:** < 1 step per minute

**Solutions:**
```bash
# Enable bf16 if supported
--bf16

# Increase batch size (if memory allows)
--per_device_train_batch_size 2
--gradient_accumulation_steps 8  # Keep batch_size * grad_accum = 16

# Use faster optimizer
--optim paged_adamw_32bit
```

---

## Useful Commands

### Run in Background
```bash
# Using nohup
nohup python qlora.py [args] > training.log 2>&1 &

# Using screen
screen -S qlora_training
python qlora.py [args]
# Press Ctrl+A, then D to detach
# Reattach with: screen -r qlora_training

# Using tmux
tmux new -s qlora
python qlora.py [args]
# Press Ctrl+B, then D to detach
# Reattach with: tmux attach -t qlora
```

### Check Training Status
```bash
# List running processes
ps aux | grep qlora.py

# Check GPU processes
nvidia-smi

# Monitor log file
tail -f training.log

# Check output directory size
du -sh output/*/
```

### Kill Training
```bash
# Find process ID
ps aux | grep qlora.py

# Kill by PID
kill -9 <PID>

# Or if using screen/tmux
screen -r qlora_training
# Press Ctrl+C
```

---

## Next Steps

### 1. Analyze Results
```bash
# Extract metrics
python -c "
import json
with open('output/baseline_experiment/trainer_state.json') as f:
    state = json.load(f)
    print('Final loss:', state['log_history'][-1]['loss'])
"
```

### 2. Run Comparison Experiments
See `EXPERIMENT_TRACKER.md` for detailed experiment plans.

Start with:
- Experiment 1A-1C: Compare quantization types
- Experiment 2A: Try different LoRA ranks

### 3. Evaluate on MMLU
```bash
# MMLU evaluation is included if you used --do_mmlu_eval
cat output/baseline_experiment/mmlu_results.json
```

### 4. Generate Sample Outputs
```bash
# Create a test prompts file
cat > test_prompts.txt << EOF
Explain quantum computing in simple terms.
Write a Python function to calculate factorial.
What is the capital of France?
EOF

# Generate responses (you'll need to adapt the generation script)
python examples/guanaco_generate.py \
    --model_name_or_path ./output/baseline_experiment/checkpoint-2000 \
    --base_model_name_or_path huggyllama/llama-7b \
    --prompts_file test_prompts.txt
```

---

## Resource Requirements

### Minimum Requirements
- GPU: 16GB VRAM (for LLaMA-7B with 4-bit quantization)
- RAM: 32GB
- Storage: 100GB free space
- Time: 4-6 hours per full experiment

### Recommended Requirements
- GPU: 24GB+ VRAM (RTX 3090, RTX 4090, A5000, A6000)
- RAM: 64GB
- Storage: 500GB SSD
- Time: Plan for 50-100 GPU hours for all experiments

### Model Size vs GPU Memory

| Model Size | 4-bit | 8-bit | Full (fp16) | MMLU Accuracy |
|------------|-------|-------|-------------|---------------|
| 7B | ~6GB | ~12GB | ~28GB | ~45% |
| 13B | ~10GB | ~20GB | ~52GB | ~50% |
| 33B | ~24GB | ~48GB | ~132GB | ~55% |
| 65B | ~48GB | ~96GB | ~260GB | ~60% |

---

## Experiment Checklist

Before starting experiments, ensure:

- [ ] GPU has sufficient memory (check `nvidia-smi`)
- [ ] Enough disk space (check `df -h`)
- [ ] Dependencies installed correctly
- [ ] Wandb configured (or using `--report_to none`)
- [ ] Quick test run completed successfully
- [ ] Output directory has write permissions
- [ ] Background running setup (screen/tmux/nohup)

During experiments:

- [ ] Monitor GPU usage regularly
- [ ] Check training logs for errors
- [ ] Verify checkpoints are saving
- [ ] Track metrics in wandb/logs
- [ ] Note any anomalies or issues

After experiments:

- [ ] Verify checkpoint integrity
- [ ] Extract and save metrics
- [ ] Run evaluation scripts
- [ ] Generate sample outputs
- [ ] Backup important checkpoints
- [ ] Document results in tracker

---

## Helpful Scripts

### Batch Experiment Runner

Create `run_experiments.sh`:
```bash
#!/bin/bash

# Run multiple experiments sequentially
experiments=(
    "4 nf4 true exp_1A"
    "4 fp4 true exp_1C"
    "8 nf4 true exp_1E"
)

for exp in "${experiments[@]}"; do
    IFS=' ' read -r bits quant dq name <<< "$exp"
    
    echo "Running experiment: $name"
    python qlora.py \
        --model_name_or_path huggyllama/llama-7b \
        --output_dir ./output/$name \
        --dataset alpaca \
        --bits $bits \
        --quant_type $quant \
        $([ "$dq" == "true" ] && echo "--double_quant") \
        --lora_r 64 --lora_alpha 16 \
        --learning_rate 0.0002 --max_steps 2000 \
        --do_train --do_eval --do_mmlu_eval \
        --logging_steps 10 --save_steps 500 \
        --bf16 --report_to wandb --run_name $name
    
    echo "Completed: $name"
done
```

Make it executable:
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

### Results Extractor

Create `extract_results.py`:
```python
#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def extract_results(output_dir):
    output_path = Path(output_dir)
    state_file = output_path / "trainer_state.json"
    
    if not state_file.exists():
        print(f"No trainer_state.json found in {output_dir}")
        return
    
    with open(state_file) as f:
        state = json.load(f)
    
    print(f"\n=== Results for {output_dir} ===")
    print(f"Total steps: {state['global_step']}")
    
    # Find final training loss
    train_losses = [log for log in state['log_history'] if 'loss' in log]
    if train_losses:
        print(f"Final training loss: {train_losses[-1]['loss']:.4f}")
    
    # Find evaluation results
    eval_results = [log for log in state['log_history'] if 'eval_loss' in log]
    if eval_results:
        print(f"Final eval loss: {eval_results[-1]['eval_loss']:.4f}")
    
    # Check for MMLU results
    mmlu_file = output_path / "mmlu_results.json"
    if mmlu_file.exists():
        with open(mmlu_file) as f:
            mmlu = json.load(f)
        print(f"MMLU Accuracy: {mmlu.get('accuracy', 'N/A')}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_results(sys.argv[1])
    else:
        # Extract from all experiments
        for exp_dir in Path("output").glob("exp_*"):
            extract_results(exp_dir)
```

Usage:
```bash
chmod +x extract_results.py
./extract_results.py output/baseline_experiment
# or extract all
./extract_results.py
```

---

## Additional Resources

### Official Documentation
- QLoRA Paper: https://arxiv.org/abs/2305.14314
- HuggingFace PEFT: https://huggingface.co/docs/peft
- bitsandbytes: https://github.com/TimDettmers/bitsandbytes

### Community Resources
- QLoRA GitHub Issues: Check for known issues
- HuggingFace Forums: Community discussions
- Reddit r/LocalLLaMA: Practical tips

### Your Project Files
- `INTERMEDIARY_REPORT.md`: Complete project documentation
- `EXPERIMENT_TRACKER.md`: Detailed experiment tracking
- `qlora/README.md`: Official QLoRA documentation

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│         QLoRA Quick Reference                    │
├─────────────────────────────────────────────────┤
│ Start training:                                  │
│   python qlora.py --model_name_or_path MODEL \  │
│       --output_dir OUTPUT --dataset DATASET      │
│                                                  │
│ Monitor GPU:                                     │
│   watch -n 1 nvidia-smi                         │
│                                                  │
│ Check progress:                                  │
│   tail -f output/DIR/training.log               │
│                                                  │
│ Run in background:                               │
│   screen -S qlora                               │
│   python qlora.py [args]                        │
│   Ctrl+A, D (detach)                            │
│                                                  │
│ Kill training:                                   │
│   ps aux | grep qlora                           │
│   kill -9 PID                                   │
│                                                  │
│ Typical memory (7B):                             │
│   4-bit: ~6-8 GB                                │
│   8-bit: ~10-14 GB                              │
└─────────────────────────────────────────────────┘
```

---

**You're ready to start experimenting! 🎉**

Begin with the quick test run, then move to full experiments once everything works.

Good luck with your research!

