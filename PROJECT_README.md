# MLDM Project: QLoRA Experimental Analysis

## 📋 Project Overview

This project conducts systematic experiments on **QLoRA (Quantized Low-Rank Adaptation)**, a memory-efficient method for fine-tuning Large Language Models. The goal is to evaluate different quantization strategies, LoRA configurations, and training hyperparameters to identify optimal configurations for resource-constrained environments.

**Course**: Machine Learning and Data Mining (MLDM)  
**Topic**: Efficient Fine-tuning of Quantized Large Language Models  
**Base Model**: LLaMA-7B (primary), with potential scaling to 13B/33B

---

## 📁 Project Structure

```
MLDM-Project/
├── qlora/                          # Main QLoRA codebase
│   ├── qlora.py                    # Training script
│   ├── scripts/                    # Pre-configured experiments
│   ├── eval/                       # Evaluation utilities
│   ├── data/                       # Datasets
│   └── output/                     # Experiment results (created during training)
│
├── INTERMEDIARY_REPORT.md          # 📄 Complete project report
├── EXPERIMENT_TRACKER.md           # 📊 Experiment tracking template
├── QUICK_START_GUIDE.md            # 🚀 Getting started guide
├── RESULTS_ANALYSIS_TEMPLATE.md   # 📈 Results analysis template
├── analyze_results.py              # 🔧 Automated results analyzer
└── PROJECT_README.md               # 📖 This file
```

---

## 🎯 Research Objectives

### Primary Goals
1. **Quantization Analysis**: Compare 4-bit vs 8-bit, NF4 vs FP4, and double quantization
2. **LoRA Optimization**: Determine optimal rank, alpha, and dropout values
3. **Dataset Comparison**: Evaluate performance across Alpaca, OASST1, and others
4. **Efficiency Metrics**: Measure memory usage, training time, and cost
5. **Performance Benchmarking**: MMLU evaluation and generation quality

### Expected Outcomes
- Optimal QLoRA configuration for 7B models
- Trade-off analysis: performance vs computational efficiency
- Reproducible experimental protocols
- Best practice recommendations

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Navigate to project
cd "/home/tk-lpt-0806/Desktop/MSDS/SEM3/Machine Learning and Data Mining/MLDM-Project/qlora"

# Install dependencies
pip install -U -r requirements.txt

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Run Quick Test (10 minutes)

```bash
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/quick_test \
    --dataset alpaca \
    --bits 4 --quant_type nf4 --double_quant \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0002 --max_steps 100 \
    --do_train --bf16
```

### 3. Run Full Baseline (4-6 hours)

```bash
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/baseline \
    --dataset alpaca \
    --bits 4 --quant_type nf4 --double_quant \
    --lora_r 64 --lora_alpha 16 \
    --learning_rate 0.0002 --max_steps 2000 \
    --do_train --do_eval --do_mmlu_eval \
    --bf16 --report_to wandb
```

**See `QUICK_START_GUIDE.md` for detailed instructions and troubleshooting.**

---

## 🔬 Planned Experiments

### Experiment 1: Quantization Analysis (6 runs)
Compare quantization strategies:
- 4-bit NF4 vs FP4
- Double quantization impact
- 4-bit vs 8-bit performance

**Expected Duration**: 24-36 hours total

### Experiment 2: LoRA Optimization (15 runs)
- **Phase A**: Rank ablation (8, 16, 32, 64, 128)
- **Phase B**: Alpha scaling (8, 16, 32, 64)
- **Phase C**: Dropout study (0.0, 0.05, 0.1, 0.15, 0.2)

**Expected Duration**: 60-90 hours total

### Experiment 3: Dataset Comparison (4 runs)
Evaluate on: Alpaca, OASST1, Self-Instruct, HH-RLHF

**Expected Duration**: 16-24 hours total

### Experiment 4: Learning Rate Optimization (6 runs)
Test LR values, schedulers, and warmup ratios

**Expected Duration**: 24-36 hours total

### Experiment 5: Efficiency Analysis
Profile memory, throughput, and cost metrics

**See `EXPERIMENT_TRACKER.md` for complete experimental protocol.**

---

## 📊 Results Analysis

### Automated Analysis Tool

```bash
# Analyze all experiments
./analyze_results.py --output_dir qlora/output

# Create comparison plots
./analyze_results.py \
    --output_dir qlora/output \
    --compare_metric mmlu_accuracy \
    --save_plots results_plots/

# Plot individual loss curves
./analyze_results.py \
    --output_dir qlora/output \
    --plot_individual \
    --save_plots results_plots/
```

**Outputs:**
- `experiment_summary.md`: Markdown report
- `experiment_summary.csv`: Detailed metrics
- `results_plots/`: Visualization figures

### Manual Analysis

Use `RESULTS_ANALYSIS_TEMPLATE.md` to document findings:
- Fill in results tables
- Add visualizations
- Document conclusions
- Statistical analysis

---

## 📈 Key Metrics to Track

### Performance Metrics
- ✅ Training loss and validation loss
- ✅ MMLU accuracy (overall and per-category)
- ✅ Generation quality (perplexity, coherence)

### Efficiency Metrics
- ✅ Peak GPU memory usage
- ✅ Training throughput (samples/sec)
- ✅ Total training time
- ✅ Checkpoint size

### Model Metrics
- ✅ Trainable parameters
- ✅ Total model size
- ✅ Parameter efficiency ratio

---

## 🛠️ Utility Scripts

### Monitor Training
```bash
# Watch GPU
watch -n 1 nvidia-smi

# Monitor logs
tail -f qlora/output/baseline/training.log

# Check progress
ls -lh qlora/output/baseline/checkpoint-*
```

### Background Execution
```bash
# Using screen
screen -S qlora_exp
python qlora.py [args]
# Ctrl+A, D to detach

# Using nohup
nohup python qlora.py [args] > training.log 2>&1 &
```

### Results Extraction
```bash
# Extract specific metrics
python -c "
import json
with open('qlora/output/baseline/trainer_state.json') as f:
    state = json.load(f)
    print('Final loss:', state['log_history'][-1].get('loss'))
"
```

---

## 📚 Documentation Guide

### 1. **INTERMEDIARY_REPORT.md** (Main Report)
   - Complete project background
   - Technical overview
   - Experimental design
   - Timeline and milestones
   - **👉 Start here for understanding the project**

### 2. **QUICK_START_GUIDE.md** (Getting Started)
   - Step-by-step setup instructions
   - Quick test run
   - Common issues and solutions
   - Useful commands
   - **👉 Use this to run your first experiment**

### 3. **EXPERIMENT_TRACKER.md** (Tracking)
   - Detailed experiment configurations
   - Command templates
   - Progress tracking tables
   - Notes section
   - **👉 Use this during experimentation**

### 4. **RESULTS_ANALYSIS_TEMPLATE.md** (Analysis)
   - Results tables
   - Visualization placeholders
   - Statistical analysis sections
   - Conclusions template
   - **👉 Use this after experiments complete**

---

## 🔧 System Requirements

### Minimum (7B Model)
- **GPU**: 16GB VRAM (e.g., RTX 4000, Tesla T4)
- **RAM**: 32GB
- **Storage**: 100GB free
- **Time**: ~4-6 hours per experiment

### Recommended (7B-13B Models)
- **GPU**: 24GB VRAM (e.g., RTX 3090, RTX 4090, A5000)
- **RAM**: 64GB
- **Storage**: 500GB SSD
- **Time**: ~100 GPU hours for all experiments

### Optimal (13B-33B Models)
- **GPU**: 48GB+ VRAM (e.g., A6000, A100)
- **RAM**: 128GB
- **Storage**: 1TB SSD

---

## 🐛 Common Issues

### CUDA Out of Memory
```bash
# Reduce batch size
--per_device_train_batch_size 1
--gradient_accumulation_steps 32

# Enable gradient checkpointing
--gradient_checkpointing

# Reduce max memory
--max_memory_MB 20000
```

### Slow Training
```bash
# Use bf16 if supported
--bf16

# Increase batch size (if memory allows)
--per_device_train_batch_size 2

# Use paged optimizer
--optim paged_adamw_32bit
```

### Model Download Failures
```bash
# Set cache directory
export HF_HOME=/path/to/large/storage
export TRANSFORMERS_CACHE=$HF_HOME

# Pre-download model
huggingface-cli download huggyllama/llama-7b
```

---

## 📦 Dependencies

### Core Libraries
```
bitsandbytes==0.40.0      # Quantization
transformers==4.31.0      # Model framework
peft==0.4.0               # LoRA implementation
accelerate==0.21.0        # Multi-GPU support
torch>=2.0.0              # PyTorch
```

### Utilities
```
wandb==0.15.3             # Experiment tracking
evaluate==0.4.0           # Metrics
scikit-learn==1.2.2       # Analysis
pandas                     # Data processing
matplotlib                 # Visualization
seaborn                    # Plotting
```

---

## 📊 Expected Results

### Performance Targets (LLaMA-7B)
- **Baseline 4-bit**: ~45% MMLU accuracy
- **Memory Usage**: 6-8 GB GPU RAM
- **Training Speed**: ~0.5-1 step/sec
- **Checkpoint Size**: ~3-5 GB

### Comparison to Baselines
- **vs Full Precision**: 95%+ performance at 25% memory
- **vs 8-bit**: Similar performance at 50% memory
- **vs Zero-shot**: +10-15% MMLU improvement

---

## 🎓 Learning Outcomes

Upon completion, you will have:

1. ✅ Hands-on experience with LLM fine-tuning
2. ✅ Understanding of quantization techniques
3. ✅ Knowledge of parameter-efficient methods (LoRA)
4. ✅ Practical skills in experiment design
5. ✅ Experience with GPU optimization
6. ✅ Data analysis and visualization skills
7. ✅ Technical report writing experience

---

## 📖 References

### Key Papers
1. **QLoRA**: Dettmers et al., 2023 - [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
2. **LoRA**: Hu et al., 2021 - [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
3. **LLaMA**: Touvron et al., 2023 - [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
4. **Alpaca**: Taori et al., 2023 - [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

### Resources
- [HuggingFace PEFT Docs](https://huggingface.co/docs/peft)
- [bitsandbytes GitHub](https://github.com/TimDettmers/bitsandbytes)
- [QLoRA Repository](https://github.com/artidoro/qlora)
- [Transformers Docs](https://huggingface.co/docs/transformers)

---

## 📝 Progress Checklist

### Phase 1: Setup ⏳
- [ ] Environment configured
- [ ] Dependencies installed
- [ ] GPU verified
- [ ] Quick test successful
- [ ] Wandb configured

### Phase 2: Baseline ⏳
- [ ] Baseline experiment running
- [ ] First checkpoint saved
- [ ] Monitoring working
- [ ] Results collected

### Phase 3: Experiments ⏳
- [ ] Experiment 1 (Quantization) complete
- [ ] Experiment 2 (LoRA) complete
- [ ] Experiment 3 (Datasets) complete
- [ ] Experiment 4 (Learning Rate) complete
- [ ] Experiment 5 (Efficiency) complete

### Phase 4: Analysis ⏳
- [ ] Results extracted
- [ ] Visualizations created
- [ ] Statistical analysis done
- [ ] Templates filled

### Phase 5: Reporting ⏳
- [ ] Final report written
- [ ] Presentation prepared
- [ ] Code documented
- [ ] Repository organized

---

## 🤝 Contributing

This is an academic project. If you're collaborating:

1. Document all experiments in `EXPERIMENT_TRACKER.md`
2. Use consistent naming: `exp_<number>_<description>`
3. Commit results regularly
4. Update progress checklist
5. Share insights in team meetings

---

## 📧 Contact & Support

### Project Maintainer
- **Student**: [Your Name]
- **Course**: Machine Learning and Data Mining
- **Institution**: [Your Institution]
- **Semester**: Fall 2025

### Getting Help
1. Check `QUICK_START_GUIDE.md` for common issues
2. Review QLoRA repository issues
3. Consult HuggingFace forums
4. Ask course instructor/TA

---

## 📄 License

This project builds on:
- **QLoRA**: MIT License
- **LLaMA**: Meta AI License (access required)
- **Alpaca**: Apache 2.0

Educational use for MLDM course project.

---

## 🎯 Next Steps

1. **Read** `INTERMEDIARY_REPORT.md` for complete background
2. **Follow** `QUICK_START_GUIDE.md` to set up environment
3. **Run** baseline experiment
4. **Track** progress in `EXPERIMENT_TRACKER.md`
5. **Analyze** results with `analyze_results.py`
6. **Document** findings in `RESULTS_ANALYSIS_TEMPLATE.md`

---

## 🌟 Project Goals Summary

| Goal | Target | Status |
|------|--------|--------|
| Setup environment | 1 day | ⏳ |
| Run baseline | 1 experiment | ⏳ |
| Complete Exp 1-4 | 30+ experiments | ⏳ |
| Analyze results | All metrics | ⏳ |
| Write final report | Complete | ⏳ |
| Achieve MMLU | >45% accuracy | ⏳ |
| Memory efficiency | <10GB for 7B | ⏳ |

---

**Last Updated**: December 1, 2025  
**Version**: 1.0  
**Status**: Ready to begin experimentation

**Good luck with your research! 🚀**

