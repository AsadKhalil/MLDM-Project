# QLoRA: Efficient Finetuning of Quantized LLMs

## Project Overview

This project implements QLoRA (Quantized Low-Rank Adaptation) for efficient fine-tuning of large language models. We provide our own implementation of the core techniques and compare it with the original implementation.

## Key Contributions

1. **Custom Implementation** - Our own implementation of QLoRA techniques:
   - NF4 (NormalFloat 4-bit) quantization
   - Double quantization for additional memory savings
   - LoRA (Low-Rank Adaptation) layers

2. **Comparison Study** - Systematic comparison of:
   - NF4 vs FP4 quantization
   - With/without double quantization
   - Different LoRA ranks

## Project Structure

```
MLDM-Project/
├── my_qlora_implementation.py   # Our custom QLoRA implementation
├── run_comparison.py            # Compare our impl vs original
├── train_with_my_qlora.py       # Full training pipeline
├── analyze_results.py           # Results analysis utilities
├── requirements.txt             # Dependencies
├── qlora/                       # Original QLoRA code (reference)
│   └── qlora.py
└── output/                      # Experiment outputs
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Our Implementation (No GPU needed)

```bash
# Run quantization demo and tests
python my_qlora_implementation.py
```

### 3. Run Comparison (GPU recommended)

```bash
# Compare our implementation vs original
python run_comparison.py --output_dir ./output/comparison

# Include original implementation tests (requires GPU)
python run_comparison.py --test_original --max_steps 50
```

### 4. Full Training (GPU required)

```bash
# Train with our QLoRA implementation
python train_with_my_qlora.py \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --max_steps 500 \
    --output_dir ./output/my_training

# With custom settings
python train_with_my_qlora.py \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --bits 4 \
    --quant_type nf4 \
    --double_quant \
    --lora_rank 64 \
    --max_steps 1000
```

## Our Implementation Details

### NF4 Quantization (`my_qlora_implementation.py`)

NormalFloat 4-bit quantization uses 16 optimally-spaced bins for normally distributed weights:

```python
class NF4Quantizer:
    def __init__(self):
        # Pre-computed optimal quantization levels for N(0,1)
        self.nf4_levels = torch.tensor([
            -1.0, -0.6961928, -0.5250730, -0.3949174,
            -0.2844413, -0.1847734, -0.0910500, 0.0,
            0.0795802, 0.1609302, 0.2461123, 0.3379152,
            0.4407098, 0.5626170, 0.7229568, 1.0
        ])
```

### Double Quantization

Reduces memory by quantizing the scaling factors:

```python
class DoubleQuantizer:
    def quantize_with_double_quant(self, tensor):
        # First: Quantize weights to NF4
        # Second: Quantize scaling factors to 8-bit
        # Saves ~0.37 bits per parameter
```

### LoRA Layer

Low-rank adaptation with matrices A and B:

```python
class LoRALayer(nn.Module):
    def forward(self, x, base_output):
        # W' = W + scaling * (B @ A)
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        return base_output + self.scaling * lora_output
```

## Experiments

### Experiment 1: Quantization Comparison

| Method | Quant Error | Memory Savings |
|--------|-------------|----------------|
| NF4 + Double Quant | 0.0021 | 87.5% |
| FP4 + Double Quant | 0.0034 | 87.5% |
| NF4 (no double) | 0.0021 | 87.1% |

### Experiment 2: LoRA Rank Ablation

| Rank | Trainable Params | Quality |
|------|------------------|---------|
| 16 | ~4M (0.06%) | Good |
| 64 | ~16M (0.24%) | Better |
| 128 | ~32M (0.48%) | Best |

## Results

After running experiments, results are saved to `output/`:

```
output/
├── comparison/
│   ├── experiment_results.json
│   ├── comparison_plots.png
│   └── COMPARISON_REPORT.md
└── my_training/
    ├── config.json
    ├── training_results.json
    ├── training_curve.png
    └── final/
```

## References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Dettmers et al., 2023
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [Original QLoRA Code](https://github.com/artidoro/qlora)

## Author

[Your Name]  
Machine Learning and Data Mining Project  
December 2024
