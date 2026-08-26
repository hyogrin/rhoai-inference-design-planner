# RedHatAI/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

## Model Overview

This is a quantized (FP8) version of NVIDIA's Nemotron-3-Nano-30B-A3B model, optimized by Red Hat AI for deployment with vLLM on OpenShift AI.

## Evaluation Results

The following table shows accuracy recovery compared to the BF16 baseline:

| Benchmark | BF16 Score | FP8 Score | Recovery |
|-----------|-----------|-----------|----------|
| MMLU      | 72.1      | 71.8      | 99.6%    |
| ARC-C     | 62.5      | 62.1      | 99.4%    |
| HellaSwag | 83.2      | 82.9      | 99.6%    |
| WinoGrande| 77.8      | 77.5      | 99.6%    |
| GSM8K     | 68.4      | 67.9      | 99.3%    |

### Evaluation Details

- **Checkpoint**: `RedHatAI/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (rev: `abc123def`)
- **vLLM Version**: 0.8.4
- **Hardware**: 2x NVIDIA H100 80GB
- **Quantization**: FP8 (W8A8) via llm-compressor
- **Evaluation Framework**: lm-eval v0.4.3

### Launch Command

```bash
vllm serve RedHatAI/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

## Intended Use

This model is intended for research and development purposes. It is optimized for inference on Red Hat OpenShift AI with vLLM serving runtime.

## Limitations

- FP8 quantization may affect performance on tasks requiring high numerical precision
- Context length limited to 8192 tokens in tested configuration
- Not validated for safety-critical applications
