# QLoRA Experiment Tracker

## Quick Reference Guide

### Experiment Status Legend
- ⏳ **Planned**: Not yet started
- 🔄 **In Progress**: Currently running
- ✅ **Completed**: Finished successfully
- ❌ **Failed**: Encountered errors
- 🔁 **Rerunning**: Needs to be repeated

---

## Experiment 1: Quantization Analysis

### Overview
**Objective**: Compare different quantization strategies  
**Model**: LLaMA-7B  
**Dataset**: Alpaca  
**Duration**: ~4-6 hours per run

| ID | Bits | Quant Type | Double Quant | Status | Memory (GB) | Loss | MMLU | Notes |
|----|------|------------|--------------|--------|-------------|------|------|-------|
| 1A | 4 | nf4 | ✓ | ⏳ | - | - | - | Baseline config |
| 1B | 4 | nf4 | ✗ | ⏳ | - | - | - | No double quant |
| 1C | 4 | fp4 | ✓ | ⏳ | - | - | - | FP4 comparison |
| 1D | 4 | fp4 | ✗ | ⏳ | - | - | - | FP4 no double |
| 1E | 8 | - | ✓ | ⏳ | - | - | - | 8-bit baseline |
| 1F | 8 | - | ✗ | ⏳ | - | - | - | 8-bit no double |

### Commands

```bash
# Experiment 1A: 4-bit NF4 with double quantization (BASELINE)
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/exp_1A_4bit_nf4_dq \
    --dataset alpaca \
    --bits 4 --quant_type nf4 --double_quant \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0002 --max_steps 2000 \
    --do_train --do_eval --do_mmlu_eval \
    --logging_steps 10 --save_steps 500 \
    --report_to wandb --run_name exp_1A

# Experiment 1B: 4-bit NF4 without double quantization
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/exp_1B_4bit_nf4 \
    --dataset alpaca \
    --bits 4 --quant_type nf4 \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0002 --max_steps 2000 \
    --do_train --do_eval --do_mmlu_eval \
    --logging_steps 10 --save_steps 500 \
    --report_to wandb --run_name exp_1B

# Experiment 1C: 4-bit FP4 with double quantization
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/exp_1C_4bit_fp4_dq \
    --dataset alpaca \
    --bits 4 --quant_type fp4 --double_quant \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0002 --max_steps 2000 \
    --do_train --do_eval --do_mmlu_eval \
    --logging_steps 10 --save_steps 500 \
    --report_to wandb --run_name exp_1C

# Experiment 1D: 4-bit FP4 without double quantization
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/exp_1D_4bit_fp4 \
    --dataset alpaca \
    --bits 4 --quant_type fp4 \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0002 --max_steps 2000 \
    --do_train --do_eval --do_mmlu_eval \
    --logging_steps 10 --save_steps 500 \
    --report_to wandb --run_name exp_1D

# Experiment 1E: 8-bit with double quantization
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/exp_1E_8bit_dq \
    --dataset alpaca \
    --bits 8 --double_quant \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0002 --max_steps 2000 \
    --do_train --do_eval --do_mmlu_eval \
    --logging_steps 10 --save_steps 500 \
    --report_to wandb --run_name exp_1E

# Experiment 1F: 8-bit without double quantization
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/exp_1F_8bit \
    --dataset alpaca \
    --bits 8 \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0002 --max_steps 2000 \
    --do_train --do_eval --do_mmlu_eval \
    --logging_steps 10 --save_steps 500 \
    --report_to wandb --run_name exp_1F
```

---

## Experiment 2: LoRA Hyperparameter Optimization

### Phase 2A: LoRA Rank Ablation

| ID | lora_r | lora_alpha | Status | Memory | Loss | MMLU | Trainable Params |
|----|--------|------------|--------|--------|------|------|------------------|
| 2A-1 | 8 | 16 | ⏳ | - | - | - | ~4M |
| 2A-2 | 16 | 16 | ⏳ | - | - | - | ~8M |
| 2A-3 | 32 | 16 | ⏳ | - | - | - | ~16M |
| 2A-4 | 64 | 16 | ⏳ | - | - | - | ~32M |
| 2A-5 | 128 | 16 | ⏳ | - | - | - | ~64M |

### Commands

```bash
# Experiment 2A: LoRA Rank Ablation
for rank in 8 16 32 64 128; do
    python qlora.py \
        --model_name_or_path huggyllama/llama-7b \
        --output_dir ./output/exp_2A_rank_${rank} \
        --dataset alpaca \
        --bits 4 --quant_type nf4 --double_quant \
        --lora_r ${rank} --lora_alpha 16 \
        --learning_rate 0.0002 --max_steps 2000 \
        --do_train --do_eval --do_mmlu_eval \
        --logging_steps 10 --save_steps 500 \
        --report_to wandb --run_name exp_2A_r${rank}
done
```

### Phase 2B: LoRA Alpha Scaling

| ID | lora_r | lora_alpha | Status | Loss | MMLU | Notes |
|----|--------|------------|--------|------|------|-------|
| 2B-1 | 64 | 8 | ⏳ | - | - | Lower alpha |
| 2B-2 | 64 | 16 | ⏳ | - | - | Baseline |
| 2B-3 | 64 | 32 | ⏳ | - | - | Higher alpha |
| 2B-4 | 64 | 64 | ⏳ | - | - | Alpha = rank |

### Commands

```bash
# Experiment 2B: LoRA Alpha Scaling
for alpha in 8 16 32 64; do
    python qlora.py \
        --model_name_or_path huggyllama/llama-7b \
        --output_dir ./output/exp_2B_alpha_${alpha} \
        --dataset alpaca \
        --bits 4 --quant_type nf4 --double_quant \
        --lora_r 64 --lora_alpha ${alpha} \
        --learning_rate 0.0002 --max_steps 2000 \
        --do_train --do_eval --do_mmlu_eval \
        --logging_steps 10 --save_steps 500 \
        --report_to wandb --run_name exp_2B_a${alpha}
done
```

### Phase 2C: LoRA Dropout

| ID | Dropout | Status | Loss | MMLU | Generalization |
|----|---------|--------|------|------|----------------|
| 2C-1 | 0.0 | ⏳ | - | - | Baseline |
| 2C-2 | 0.05 | ⏳ | - | - | Light regularization |
| 2C-3 | 0.1 | ⏳ | - | - | Moderate regularization |
| 2C-4 | 0.15 | ⏳ | - | - | Strong regularization |
| 2C-5 | 0.2 | ⏳ | - | - | Very strong |

### Commands

```bash
# Experiment 2C: LoRA Dropout
for dropout in 0.0 0.05 0.1 0.15 0.2; do
    python qlora.py \
        --model_name_or_path huggyllama/llama-7b \
        --output_dir ./output/exp_2C_dropout_${dropout} \
        --dataset alpaca \
        --bits 4 --quant_type nf4 --double_quant \
        --lora_r 64 --lora_alpha 16 --lora_dropout ${dropout} \
        --learning_rate 0.0002 --max_steps 2000 \
        --do_train --do_eval --do_mmlu_eval \
        --logging_steps 10 --save_steps 500 \
        --report_to wandb --run_name exp_2C_d${dropout}
done
```

---

## Experiment 3: Dataset Comparison

| ID | Dataset | Size | Status | Loss | MMLU | Conversational | Notes |
|----|---------|------|--------|------|------|----------------|-------|
| 3A | alpaca | 52K | ⏳ | - | - | - | Baseline |
| 3B | oasst1 | 84K | ⏳ | - | - | - | Guanaco dataset |
| 3C | self-instruct | 82K | ⏳ | - | - | - | Auto-generated |
| 3D | hh-rlhf | 160K | ⏳ | - | - | - | Human feedback |

### Commands

```bash
# Experiment 3: Dataset Comparison
for dataset in alpaca oasst1; do
    python qlora.py \
        --model_name_or_path huggyllama/llama-7b \
        --output_dir ./output/exp_3_${dataset} \
        --dataset ${dataset} \
        --bits 4 --quant_type nf4 --double_quant \
        --lora_r 64 --lora_alpha 16 \
        --learning_rate 0.0002 --max_steps 2000 \
        --do_train --do_eval --do_mmlu_eval \
        --logging_steps 10 --save_steps 500 \
        --report_to wandb --run_name exp_3_${dataset}
done
```

---

## Experiment 4: Learning Rate Optimization

| ID | LR | Scheduler | Warmup | Status | Convergence | Final Loss | MMLU |
|----|----|-----------|----|--------|-------------|------------|------|
| 4A | 2e-4 | constant | 0.03 | ⏳ | - | - | - |
| 4B | 1e-4 | constant | 0.03 | ⏳ | - | - | - |
| 4C | 5e-4 | constant | 0.03 | ⏳ | - | - | - |
| 4D | 2e-4 | cosine | 0.03 | ⏳ | - | - | - |
| 4E | 2e-4 | linear | 0.03 | ⏳ | - | - | - |
| 4F | 2e-4 | constant | 0.1 | ⏳ | - | - | - |

### Commands

```bash
# Experiment 4A: Baseline LR
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/exp_4A_lr_2e4_constant \
    --dataset alpaca \
    --bits 4 --quant_type nf4 --double_quant \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0002 --lr_scheduler_type constant --warmup_ratio 0.03 \
    --max_steps 2000 --do_train --do_eval --do_mmlu_eval \
    --logging_steps 10 --save_steps 500 \
    --report_to wandb --run_name exp_4A

# Experiment 4B: Lower LR
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/exp_4B_lr_1e4_constant \
    --dataset alpaca \
    --bits 4 --quant_type nf4 --double_quant \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0001 --lr_scheduler_type constant --warmup_ratio 0.03 \
    --max_steps 2000 --do_train --do_eval --do_mmlu_eval \
    --logging_steps 10 --save_steps 500 \
    --report_to wandb --run_name exp_4B

# Experiment 4C: Higher LR
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/exp_4C_lr_5e4_constant \
    --dataset alpaca \
    --bits 4 --quant_type nf4 --double_quant \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0005 --lr_scheduler_type constant --warmup_ratio 0.03 \
    --max_steps 2000 --do_train --do_eval --do_mmlu_eval \
    --logging_steps 10 --save_steps 500 \
    --report_to wandb --run_name exp_4C

# Experiment 4D: Cosine scheduler
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/exp_4D_lr_2e4_cosine \
    --dataset alpaca \
    --bits 4 --quant_type nf4 --double_quant \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0002 --lr_scheduler_type cosine --warmup_ratio 0.03 \
    --max_steps 2000 --do_train --do_eval --do_mmlu_eval \
    --logging_steps 10 --save_steps 500 \
    --report_to wandb --run_name exp_4D

# Experiment 4E: Linear scheduler
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/exp_4E_lr_2e4_linear \
    --dataset alpaca \
    --bits 4 --quant_type nf4 --double_quant \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0002 --lr_scheduler_type linear --warmup_ratio 0.03 \
    --max_steps 2000 --do_train --do_eval --do_mmlu_eval \
    --logging_steps 10 --save_steps 500 \
    --report_to wandb --run_name exp_4E

# Experiment 4F: Higher warmup
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/exp_4F_lr_2e4_warmup01 \
    --dataset alpaca \
    --bits 4 --quant_type nf4 --double_quant \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0002 --lr_scheduler_type constant --warmup_ratio 0.1 \
    --max_steps 2000 --do_train --do_eval --do_mmlu_eval \
    --logging_steps 10 --save_steps 500 \
    --report_to wandb --run_name exp_4F
```

---

## Progress Tracking

### Week 1: Setup & Baseline
- [ ] Environment setup complete
- [ ] Dependencies installed
- [ ] Base model downloaded
- [ ] Alpaca dataset prepared
- [ ] Baseline experiment (1A) running
- [ ] WandB configured

### Week 2: Quantization Experiments
- [ ] Experiments 1A-1F completed
- [ ] Results collected and analyzed
- [ ] Best quantization strategy identified

### Week 3: LoRA Optimization
- [ ] Phase 2A (rank) completed
- [ ] Phase 2B (alpha) completed
- [ ] Phase 2C (dropout) completed
- [ ] Optimal LoRA config determined

### Week 4: Datasets & Learning Rate
- [ ] Experiment 3 (datasets) completed
- [ ] Experiment 4 (LR) completed
- [ ] Cross-evaluation performed

### Week 5-6: Analysis & Reporting
- [ ] All experiments completed
- [ ] Visualizations created
- [ ] Statistical analysis done
- [ ] Final report written

---

## Quick Tips

### Monitoring Training
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f output/exp_1A_4bit_nf4_dq/training_log.txt

# Check checkpoint disk usage
du -sh output/*/checkpoint-*
```

### Handling Interruptions
```bash
# Resume from checkpoint
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --resume_from_checkpoint output/exp_1A/checkpoint-1000 \
    --output_dir ./output/exp_1A_4bit_nf4_dq \
    # ... other parameters
```

### Batch Processing
```bash
# Run multiple experiments sequentially
./run_all_experiments.sh

# Or use screen/tmux for background running
screen -S qlora_exp1
python qlora.py ...
# Ctrl+A, D to detach
```

### Memory Issues
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`
- Reduce `max_memory_MB`

---

## Data Collection Checklist

For each experiment, ensure you collect:
- ✅ Training loss curve
- ✅ Validation loss
- ✅ MMLU accuracy (overall and per-category)
- ✅ Peak GPU memory usage
- ✅ Training time (total and per-step)
- ✅ Number of trainable parameters
- ✅ Final checkpoint size
- ✅ Sample generations
- ✅ WandB run URL

---

## Notes Section

### Experiment 1A Notes
Date: __________
Status: ⏳
Issues: 
Results:

### Experiment 1B Notes
Date: __________
Status: ⏳
Issues:
Results:

### Experiment 1C Notes
Date: __________
Status: ⏳
Issues:
Results:

(Continue for all experiments...)

