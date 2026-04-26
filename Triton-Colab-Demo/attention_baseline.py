import argparse
import torch
import torch.nn.functional as F

from utils import cuda_time_ms


def attention_pytorch(q, k, v):
    """
    Standard attention baseline:
    scores = QK^T / sqrt(d)
    probs = softmax(scores)
    output = probs V

    This materializes the full attention matrix [B, H, S, S].
    FlashAttention avoids materializing this full matrix by tiling and fusing operations.
    """
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5)
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def attention_sdpa(q, k, v):
    """PyTorch 2.0 scaled_dot_product_attention — uses FlashAttention when available."""
    return F.scaled_dot_product_attention(q, k, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    print("GPU:", torch.cuda.get_device_name(0))

    shape = (args.batch, args.heads, args.seq_len, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    torch.cuda.reset_peak_memory_stats()
    pt_ms = cuda_time_ms(lambda: attention_pytorch(q, k, v), iters=args.iters)
    peak_mem_pt_gb = torch.cuda.max_memory_allocated() / 1024**3

    torch.cuda.reset_peak_memory_stats()
    sdpa_ms = cuda_time_ms(lambda: attention_sdpa(q, k, v), iters=args.iters)
    peak_mem_sdpa_gb = torch.cuda.max_memory_allocated() / 1024**3

    attention_matrix_elements = args.batch * args.heads * args.seq_len * args.seq_len
    attention_matrix_mb = attention_matrix_elements * 2 / 1024**2

    print("\n=== Attention Baseline ===")
    print(f"Shape: B={args.batch}, H={args.heads}, S={args.seq_len}, D={args.head_dim}")
    print(f"Materialized attention matrix size: ~{attention_matrix_mb:.2f} MB")
    print(f"\nPyTorch naive attention:  {pt_ms:.4f} ms  |  peak mem: {peak_mem_pt_gb:.3f} GB")
    print(f"PyTorch SDPA (Flash):     {sdpa_ms:.4f} ms  |  peak mem: {peak_mem_sdpa_gb:.3f} GB")
    print(f"Speedup (SDPA / naive):   {pt_ms / sdpa_ms:.2f}x")
    print("\nFlashAttention idea:")
    print("- Do not materialize full [B, H, S, S] attention matrix")
    print("- Tile Q/K/V blocks")
    print("- Fuse QK^T, softmax, and V multiplication")
    print("- Reduce HBM memory traffic")


if __name__ == "__main__":
    main()
