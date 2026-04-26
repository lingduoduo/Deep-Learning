# Triton + FlashAttention-Style Colab Demo

A clean, runnable Google Colab demo for CUDA / high-performance kernel optimization using Triton.

This project implements:

- PyTorch baseline softmax
- Triton fused row-wise softmax kernel
- Attention-style benchmark setup
- Latency comparison
- Bandwidth-style analysis
- Colab notebook

This is not a full FlashAttention implementation. It is a FlashAttention-inspired demo that shows the core idea behind fused GPU kernels: reduce memory movement and improve execution time.

## Colab Setup

1. Open Google Colab
2. Runtime → Change runtime type → GPU
3. Upload this zip file
4. Run:

```bash
!unzip -q -o triton-flashattention-colab-demo.zip
%cd triton-flashattention-colab-demo
!pip install -r requirements.txt
```

5. Run benchmark:

```bash
!python benchmark_softmax.py --rows 4096 --cols 1024 --warmup 10 --iters 50
```

6. Run attention-style benchmark:

```bash
!python attention_baseline.py --batch 4 --heads 8 --seq_len 512 --head_dim 64 --iters 20
```
