# Intermediary Report: QLoRA - Efficient Fine-tuning of Quantized Large Language Models

## Project Information

**Course**: Machine Learning and Data Mining (MLDM)  
**Project Title**: Experimental Analysis of QLoRA for Efficient LLM Fine-tuning  
**Date**: December 1, 2025  
**Repository**: MLDM-Project/qlora

---

## 1. Executive Summary

This intermediary report presents the progress and planned experiments for investigating QLoRA (Quantized Low-Rank Adaptation), a memory-efficient approach for fine-tuning Large Language Models (LLMs). QLoRA enables fine-tuning of models with up to 65 billion parameters on consumer-grade GPUs through 4-bit quantization combined with Low-Rank Adapters (LoRA). This project aims to conduct systematic experiments to evaluate the effectiveness of various quantization strategies, LoRA configurations, and training hyperparameters on model performance and computational efficiency.

---

## 2. Introduction

### 2.1 Background

Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse natural language processing tasks. However, their massive size (billions of parameters) presents significant challenges for fine-tuning:

- **Memory Requirements**: Standard fine-tuning of a 65B parameter model requires over 780GB of GPU memory
- **Computational Cost**: Full fine-tuning is prohibitively expensive for most researchers and organizations
- **Accessibility**: Limited access to high-end hardware restricts LLM research to well-funded institutions

### 2.2 QLoRA Solution

QLoRA addresses these challenges through three key innovations:

1. **4-bit NormalFloat (NF4)**: A novel quantization data type optimized for normally distributed neural network weights
2. **Double Quantization**: Further memory reduction by quantizing the quantization constants themselves
3. **Paged Optimizers**: Dynamic memory management to handle optimizer state memory spikes

These techniques enable:
- Fine-tuning 65B models on a single 48GB GPU
- Preservation of 16-bit fine-tuning performance
- 4x-8x memory reduction compared to standard approaches

### 2.3 Research Significance

Understanding QLoRA's effectiveness across different configurations is crucial for:
- Democratizing access to LLM research
- Optimizing resource utilization in model fine-tuning
- Establishing best practices for parameter-efficient fine-tuning
- Balancing model performance with computational constraints

---

## 3. Project Objectives

### 3.1 Primary Objectives

1. **Quantization Analysis**: Systematically evaluate different quantization strategies (4-bit vs 8-bit, NF4 vs FP4, double quantization effects)
2. **LoRA Optimization**: Determine optimal LoRA hyperparameters (rank, alpha, dropout) for different model sizes
3. **Dataset Comparison**: Assess model performance across multiple instruction-following datasets
4. **Efficiency Metrics**: Measure memory usage, training time, and inference speed under various configurations
5. **Performance Benchmarking**: Evaluate fine-tuned models on standardized benchmarks (MMLU)

### 3.2 Secondary Objectives

1. Document reproducible experimental protocols
2. Analyze trade-offs between model quality and computational efficiency
3. Identify optimal configurations for resource-constrained environments
4. Generate insights for future parameter-efficient fine-tuning research

---

## 4. Technical Overview

### 4.1 Codebase Architecture

The QLoRA repository is structured as follows:

```
qlora/
├── qlora.py                 # Main training script (842 lines)
├── requirements.txt         # Dependencies
├── scripts/                 # Pre-configured training scripts
│   ├── finetune_guanaco_7b.sh
│   ├── finetune_guanaco_13b.sh
│   ├── finetune_guanaco_33b.sh
│   ├── finetune_guanaco_65b.sh
│   └── finetune.sh
├── eval/                    # Evaluation scripts and results
│   ├── eval_gpt_review.py
│   ├── generations/         # Pre-generated model outputs
│   ├── ratings-gpt4/        # GPT-4 evaluation ratings
│   └── ratings-human/       # Human evaluation data
├── data/                    # Dataset storage
│   └── mmlu/               # MMLU benchmark data
└── examples/               # Demo notebooks and scripts
```

### 4.2 Core Components

#### 4.2.1 Model Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name_or_path` | `EleutherAI/pythia-12b` | Base model to fine-tune |
| `trust_remote_code` | `False` | Enable custom model code |
| `use_auth_token` | `False` | HuggingFace authentication |

#### 4.2.2 Quantization Configuration

| Parameter | Default | Options | Impact |
|-----------|---------|---------|--------|
| `bits` | 4 | 4, 8, 16 | Quantization precision |
| `quant_type` | `nf4` | `nf4`, `fp4` | Quantization data type |
| `double_quant` | `True` | Boolean | Secondary quantization |
| `bnb_4bit_compute_dtype` | `bfloat16` | `float16`, `bfloat16` | Computation precision |

#### 4.2.3 LoRA Parameters

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `lora_r` | 64 | 8-256 | Adapter rank dimension |
| `lora_alpha` | 16 | 8-128 | Scaling parameter |
| `lora_dropout` | 0.0 | 0.0-0.3 | Regularization |
| `lora_modules` | `all` | `all`, `specific` | Target layers |

#### 4.2.4 Training Hyperparameters

| Parameter | Default | Typical Range |
|-----------|---------|---------------|
| `learning_rate` | 0.0002 | 1e-5 to 5e-4 |
| `per_device_train_batch_size` | 1 | 1-8 |
| `gradient_accumulation_steps` | 16 | 4-32 |
| `max_steps` | 10000 | 1000-50000 |
| `warmup_ratio` | 0.03 | 0.01-0.1 |
| `max_grad_norm` | 0.3 | 0.1-1.0 |

### 4.3 Supported Datasets

The codebase supports multiple instruction-following datasets:

1. **Alpaca**: Stanford's instruction-following dataset
2. **OASST1** (OpenAssistant): Community-generated conversational data
3. **Self-Instruct**: Automatically generated instructions
4. **HH-RLHF**: Anthropic's human preference dataset
5. **Custom datasets**: JSON/JSONL format support

### 4.4 Evaluation Framework

**Automatic Evaluation:**
- MMLU (Massive Multitask Language Understanding): 57 academic tasks
- Zero-shot and few-shot evaluation modes
- Accuracy metrics across subject categories

**Qualitative Evaluation:**
- GPT-4 based comparative assessment
- Human evaluation protocols
- Generation quality metrics

### 4.5 Technical Dependencies

```
bitsandbytes==0.40.0      # Quantization backend
transformers==4.31.0      # Model implementation
peft==0.4.0               # LoRA implementation
accelerate==0.21.0        # Multi-GPU support
einops==0.6.1            # Tensor operations
evaluate==0.4.0          # Evaluation metrics
scikit-learn==1.2.2      # ML utilities
wandb==0.15.3            # Experiment tracking
```

---

## 5. Current Progress

### 5.1 Repository Analysis Completed

✅ **Code Structure Understanding**
- Analyzed main training script (`qlora.py` - 842 lines)
- Reviewed configuration dataclasses for all parameter categories
- Examined pre-configured training scripts for different model sizes

✅ **Component Identification**
- Model loading with quantization (`get_accelerate_model`)
- LoRA configuration and application
- Dataset processing pipelines
- Training loop implementation (Seq2SeqTrainer)
- Evaluation modules (MMLU, generation)

✅ **Experimental Design Framework**
- Identified key hyperparameters for ablation studies
- Mapped dataset options and formats
- Documented evaluation methodologies
- Reviewed baseline configurations from paper

### 5.2 Experimental Infrastructure Assessment

**Available Resources:**
- Pre-configured scripts for 7B, 13B, 33B, 65B models
- MMLU benchmark dataset (zero-shot and few-shot)
- Evaluation scripts for automated assessment
- Generation comparison utilities

**Requirements Identified:**
- GPU with minimum 24GB VRAM for 7B models
- 48GB+ VRAM for larger models
- Storage for model checkpoints (~10-20GB per experiment)
- Compute time: 4-24 hours per training run depending on model size

---

## 6. Planned Experimental Design

### 6.1 Experiment 1: Quantization Strategy Analysis

**Objective**: Evaluate the impact of different quantization approaches on model performance and memory efficiency.

**Variables:**
- Quantization bits: 4-bit vs 8-bit
- Quantization type: NF4 vs FP4
- Double quantization: Enabled vs Disabled

**Configuration Matrix:**

| Experiment | Bits | Quant Type | Double Quant | Expected Memory |
|------------|------|------------|--------------|-----------------|
| 1A | 4 | nf4 | True | ~6GB (7B model) |
| 1B | 4 | nf4 | False | ~8GB |
| 1C | 4 | fp4 | True | ~6GB |
| 1D | 4 | fp4 | False | ~8GB |
| 1E | 8 | - | True | ~12GB |
| 1F | 8 | - | False | ~16GB |

**Fixed Parameters:**
- Model: LLaMA-7B
- Dataset: Alpaca
- LoRA rank: 64
- Learning rate: 0.0002
- Max steps: 2000

**Metrics to Collect:**
- Peak GPU memory usage
- Training time per step
- Final validation loss
- MMLU accuracy
- Generation quality (perplexity)

### 6.2 Experiment 2: LoRA Hyperparameter Optimization

**Objective**: Determine optimal LoRA configuration for parameter-efficient fine-tuning.

**Phase 2A: LoRA Rank Ablation**

| Experiment | lora_r | lora_alpha | Trainable Params | Expected Performance |
|------------|--------|------------|------------------|----------------------|
| 2A-1 | 8 | 16 | ~4M | Lower |
| 2A-2 | 16 | 16 | ~8M | Moderate |
| 2A-3 | 32 | 16 | ~16M | Good |
| 2A-4 | 64 | 16 | ~32M | Best |
| 2A-5 | 128 | 16 | ~64M | Diminishing returns |

**Phase 2B: LoRA Alpha Scaling**

Test alpha values: [8, 16, 32, 64] with fixed rank=64

**Phase 2C: LoRA Dropout Regularization**

Test dropout values: [0.0, 0.05, 0.1, 0.15, 0.2] with optimal rank and alpha

**Fixed Parameters:**
- Model: LLaMA-7B
- Dataset: Alpaca
- Quantization: 4-bit NF4 with double quantization
- Max steps: 2000

**Metrics:**
- Training loss curve
- Validation loss
- MMLU accuracy
- Parameter count (trainable vs total)
- Overfitting indicators

### 6.3 Experiment 3: Dataset Comparison Study

**Objective**: Assess how different instruction datasets affect model capabilities.

**Datasets to Compare:**

| Dataset | Size | Domain | Instruction Type |
|---------|------|--------|------------------|
| Alpaca | ~52K | General | Task instructions |
| OASST1 | ~84K | Conversational | Multi-turn dialogue |
| Self-Instruct | ~82K | Diverse | Auto-generated |
| HH-RLHF | ~160K | Safety-focused | Human feedback |

**Evaluation Approach:**
- Train separate models on each dataset
- Evaluate on held-out test sets from each domain
- Cross-evaluate (e.g., Alpaca-trained on OASST1 test)
- Qualitative generation comparison

**Fixed Parameters:**
- Model: LLaMA-7B
- Quantization: 4-bit NF4 with double quantization
- LoRA: Optimal configuration from Experiment 2
- Training steps: Normalized by dataset size

### 6.4 Experiment 4: Learning Rate and Schedule Optimization

**Objective**: Find optimal training dynamics for QLoRA.

**Variables:**
- Learning rate: [1e-4, 2e-4, 5e-4, 1e-3]
- LR scheduler: [constant, linear, cosine]
- Warmup ratio: [0.01, 0.03, 0.05, 0.1]

**Configuration Matrix:**

| Experiment | LR | Scheduler | Warmup | Expected Convergence |
|------------|----|-----------|----|---------------------|
| 4A | 2e-4 | constant | 0.03 | Stable (baseline) |
| 4B | 1e-4 | constant | 0.03 | Slower, more stable |
| 4C | 5e-4 | constant | 0.03 | Faster, less stable |
| 4D | 2e-4 | cosine | 0.03 | Better final loss |
| 4E | 2e-4 | linear | 0.03 | Gradual decay |
| 4F | 2e-4 | constant | 0.1 | Longer warmup |

**Metrics:**
- Loss curves (training and validation)
- Convergence speed (steps to target loss)
- Final model performance
- Training stability (loss variance)

### 6.5 Experiment 5: Computational Efficiency Analysis

**Objective**: Characterize resource requirements and training efficiency.

**Measurements:**
- GPU memory usage over training
- Training throughput (samples/second)
- Gradient accumulation impact
- Batch size scaling
- Multi-GPU efficiency (if available)

**Profiling Points:**
- Model loading time
- Forward pass time
- Backward pass time
- Optimizer step time
- Checkpoint saving time

### 6.6 Experiment 6: Model Scaling Study (Resource Permitting)

**Objective**: Analyze how QLoRA scales to larger models.

**Model Sizes to Test:**
- 7B parameters (baseline)
- 13B parameters
- 30B+ parameters (if resources allow)

**Analysis:**
- Memory scaling curves
- Performance vs. size trade-offs
- Optimal LoRA rank per model size
- Training time scaling

---

## 7. Evaluation Methodology

### 7.1 Quantitative Metrics

**Performance Metrics:**
- **MMLU Accuracy**: Overall and per-category scores
- **Perplexity**: On held-out validation sets
- **Loss Values**: Training and validation loss curves

**Efficiency Metrics:**
- **Peak Memory Usage**: Maximum GPU memory consumed
- **Average Memory Usage**: Typical memory footprint
- **Training Time**: Total time and time per step
- **Throughput**: Samples processed per second
- **Parameter Efficiency**: Trainable parameters / total parameters

### 7.2 Qualitative Evaluation

**Generation Quality Assessment:**
1. Sample prompts from multiple categories:
   - General knowledge questions
   - Reasoning tasks
   - Creative writing
   - Code generation
   - Mathematical problems

2. Evaluation criteria:
   - Coherence and fluency
   - Factual accuracy
   - Instruction following
   - Response relevance
   - Format compliance

### 7.3 Comparative Analysis

**Baseline Comparisons:**
- Unquantized LoRA fine-tuning (if resources permit)
- Published QLoRA results from paper
- Zero-shot base model performance

**Statistical Testing:**
- Multiple runs with different seeds
- Confidence intervals for key metrics
- Significance testing for performance differences

---

## 8. Expected Outcomes and Hypotheses

### 8.1 Quantization Experiments

**Hypothesis 1**: NF4 quantization will outperform FP4 due to optimal information distribution for normal weights.

**Hypothesis 2**: Double quantization will reduce memory by 10-15% with minimal (<1%) performance degradation.

**Hypothesis 3**: 4-bit quantization will achieve 95%+ of 8-bit performance while using 50% less memory.

### 8.2 LoRA Optimization

**Hypothesis 4**: LoRA rank 32-64 will provide optimal trade-off between parameter efficiency and performance.

**Hypothesis 5**: Higher LoRA ranks show diminishing returns beyond r=64 for 7B models.

**Hypothesis 6**: Small dropout (0.05-0.1) will improve generalization without hurting convergence.

### 8.3 Dataset Comparison

**Hypothesis 7**: OASST1 (Guanaco) will produce better conversational models than Alpaca.

**Hypothesis 8**: Models show strong performance on in-domain tasks but moderate cross-domain transfer.

### 8.4 Training Dynamics

**Hypothesis 9**: Constant learning rate schedule performs comparably to cosine schedule for QLoRA.

**Hypothesis 10**: Optimal learning rate for 7B models is in the range [1e-4, 2e-4].

## 10. Risk Assessment and Mitigation

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Insufficient GPU memory | Medium | High | Start with 7B model, use gradient checkpointing |
| Training instability | Medium | Medium | Multiple seeds, careful LR tuning, monitor loss |
| Long training times | High | Medium | Reduce max_steps for initial experiments, use efficient batch sizes |
| CUDA/dependency errors | Medium | High | Use Docker container, pin dependency versions |
| Checkpoint corruption | Low | High | Regular backups, multiple checkpoint slots |

### 10.2 Resource Constraints

**Computational Limitations:**
- Training runs may take 4-24 hours each
- Total compute budget: ~500 GPU hours estimated
- Storage requirements: ~500GB for checkpoints and data

**Mitigation Strategies:**
- Prioritize most impactful experiments
- Use smaller max_steps for preliminary runs
- Efficient checkpoint management (save_total_limit)
- Parallel experimentation where possible

### 10.3 Timeline Risks

**Potential Delays:**
- Extended training times
- Need for additional experiment iterations
- Hardware availability issues

**Contingency Plans:**
- Focus on core experiments (1, 2, 3) as priority
- Reduce experiment scope if needed
- Use pre-trained Guanaco models for comparison


## 12. Preliminary Observations

### 12.1 Code Quality Assessment

**Strengths:**
- Well-structured with clear separation of concerns
- Comprehensive command-line arguments
- Integration with HuggingFace ecosystem
- Built-in evaluation framework

**Areas for Improvement:**
- Limited inline documentation
- Hard-coded constants in some sections
- Could benefit from modular refactoring

### 12.2 Documentation Review

**Available Resources:**
- Comprehensive README with examples
- Pre-configured training scripts
- Colab notebooks for demos
- Evaluation protocols and baselines

**Gap Analysis:**
- Limited hyperparameter tuning guidelines
- Sparse documentation on dataset formats
- Minimal troubleshooting guides

---

## 13. Related Work and Context

### 13.1 Key Papers

1. **QLoRA** (Dettmers et al., 2023): Original paper introducing the method
2. **LoRA** (Hu et al., 2021): Low-Rank Adaptation foundation
3. **LLaMA** (Touvron et al., 2023): Base model family
4. **Alpaca** (Taori et al., 2023): Instruction fine-tuning dataset

### 13.2 Competing Approaches

- **Full Fine-tuning**: Highest quality but extremely expensive
- **Adapter Layers**: Similar efficiency to LoRA but different architecture
- **Prompt Tuning**: Even more efficient but limited expressiveness
- **8-bit Training**: More memory than QLoRA, similar performance

### 13.3 Research Contributions

This project will contribute:
- Systematic ablation studies of QLoRA hyperparameters
- Reproducible experimental protocols
- Resource requirement characterization
- Best practice recommendations for practitioners

---

## 14. Conclusion

This intermediary report establishes a comprehensive framework for investigating QLoRA's efficiency and effectiveness in fine-tuning large language models. The project is well-positioned to deliver valuable insights through systematic experimentation across quantization strategies, LoRA configurations, datasets, and training dynamics.

### 14.1 Current Status

- ✅ Repository analyzed and understood
- ✅ Experimental design completed
- ✅ Success criteria defined
- ⏳ Ready to begin implementation

This research will provide:
- **Practical Guidelines**: Clear recommendations for QLoRA hyperparameters
- **Efficiency Metrics**: Detailed resource requirement characterization
- **Performance Benchmarks**: Comparative analysis across configurations
- **Reproducible Protocols**: Open experimental framework for future research

The findings will benefit researchers and practitioners seeking to leverage QLoRA for efficient LLM fine-tuning in resource-constrained environments.

---

## 15. References

1. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv preprint arXiv:2305.14314.

2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.

3. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Lample, G. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2302.13971.

4. Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., ... & Hashimoto, T. B. (2023). Stanford Alpaca: An Instruction-following LLaMA model.

5. Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. arXiv preprint arXiv:2208.07339.

6. HuggingFace PEFT Documentation: https://huggingface.co/docs/peft

7. BitsAndBytes Library: https://github.com/TimDettmers/bitsandbytes

---

## Appendix A: Command Line Examples

### A.1 Baseline Training Command

```bash
python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output/baseline \
    --dataset alpaca \
    --do_train True \
    --do_eval True \
    --do_mmlu_eval True \
    --bits 4 \
    --quant_type nf4 \
    --double_quant \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --learning_rate 0.0002 \
    --max_steps 2000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --report_to wandb
```

### A.2 Quantization Comparison

```bash
# 4-bit NF4 with double quantization
python qlora.py --bits 4 --quant_type nf4 --double_quant \
    --output_dir ./output/4bit_nf4_dq

# 4-bit FP4 with double quantization
python qlora.py --bits 4 --quant_type fp4 --double_quant \
    --output_dir ./output/4bit_fp4_dq

# 8-bit quantization
python qlora.py --bits 8 \
    --output_dir ./output/8bit
```

### A.3 LoRA Rank Sweep

```bash
for rank in 8 16 32 64 128; do
    python qlora.py \
        --lora_r $rank \
        --lora_alpha $rank \
        --output_dir ./output/lora_r_${rank}
done
```

---

## Appendix B: Expected Output Structure

```
output/
├── experiment_1_quantization/
│   ├── 4bit_nf4_dq/
│   │   ├── checkpoint-500/
│   │   ├── checkpoint-1000/
│   │   ├── training_logs.json
│   │   └── eval_results.json
│   ├── 4bit_fp4_dq/
│   └── 8bit/
├── experiment_2_lora/
│   ├── rank_8/
│   ├── rank_16/
│   ├── rank_32/
│   ├── rank_64/
│   └── rank_128/
├── experiment_3_datasets/
│   ├── alpaca/
│   ├── oasst1/
│   └── self_instruct/
└── results_summary/
    ├── metrics_comparison.csv
    ├── memory_usage.png
    └── performance_curves.png
```

---

## Appendix C: Monitoring and Logging

### C.1 Key Metrics to Track

```python
# Training metrics
- loss: Training loss per step
- learning_rate: Current learning rate
- epoch: Training epoch
- grad_norm: Gradient norm (for stability monitoring)

# Evaluation metrics
- eval_loss: Validation loss
- eval_accuracy: MMLU accuracy (if enabled)
- eval_runtime: Evaluation time

# System metrics
- gpu_memory_allocated: Current GPU memory
- gpu_memory_reserved: Reserved GPU memory
- samples_per_second: Training throughput
```

### C.2 Weights & Biases Configuration

```python
# Initialize tracking
import wandb

wandb.init(
    project="qlora-experiments",
    name=f"exp_{experiment_id}",
    config={
        "model": "llama-7b",
        "quantization": "4bit-nf4",
        "lora_rank": 64,
        "dataset": "alpaca"
    }
)
```

---

**Report Prepared By**: MLDM Student  
**Date**: December 1, 2025  
**Version**: 1.0 (Intermediary Report)

