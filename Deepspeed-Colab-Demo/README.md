# DeepSpeed Colab Demo

A clean, Google Colab–friendly demo for testing LLM training optimization with:

- Baseline PyTorch training
- DeepSpeed ZeRO-2 / ZeRO-3 configs
- FP16 mixed precision
- Gradient accumulation
- Throughput and GPU memory benchmarking

This project is intentionally small so it can run on a single free/paid Colab GPU.

## Recommended Colab Setup

1. Open Google Colab
2. Runtime → Change runtime type → GPU
3. Upload this repo zip or clone your GitHub repo
4. Run:

```bash
!pip install -r requirements.txt
```

5. Run baseline:

```bash
!python train_baseline.py --model distilgpt2 --max_steps 20 --batch_size 2 --block_size 128
```

6. Run DeepSpeed ZeRO-2:

```bash
!deepspeed --num_gpus=1 train_deepspeed.py \
  --model distilgpt2 \
  --max_steps 20 \
  --batch_size 2 \
  --block_size 128 \
  --deepspeed_config ds_zero2_config.json
```

7. Run DeepSpeed ZeRO-3:

```bash
!deepspeed --num_gpus=1 train_deepspeed.py \
  --model distilgpt2 \
  --max_steps 20 \
  --batch_size 2 \
  --block_size 128 \
  --deepspeed_config ds_zero3_config.json
```

## Metrics Logged

Each run prints:

- step loss
- tokens/sec
- peak GPU memory in GB
- total training time

## Resume Bullet

Built a reproducible Google Colab benchmark for LLM training optimization using PyTorch and DeepSpeed ZeRO, comparing baseline training with ZeRO-2/ZeRO-3, mixed precision, and gradient accumulation while tracking tokens/sec, peak GPU memory, and training loss.
