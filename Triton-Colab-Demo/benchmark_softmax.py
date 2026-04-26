import argparse
import torch
import torch.nn.functional as F

from triton_softmax import triton_softmax
from utils import cuda_time_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. In Colab, use Runtime -> Change runtime type -> GPU.")

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Tensor shape: [{args.rows}, {args.cols}]")

    x = torch.randn(args.rows, args.cols, device="cuda", dtype=torch.float32)

    y_pt = F.softmax(x, dim=-1)
    y_tri = triton_softmax(x)

    max_error = (y_pt - y_tri).abs().max().item()
    print(f"Correctness max abs error: {max_error:.6e}")

    pt_ms = cuda_time_ms(lambda: F.softmax(x, dim=-1), args.warmup, args.iters)
    tri_ms = cuda_time_ms(lambda: triton_softmax(x), args.warmup, args.iters)

    speedup = pt_ms / tri_ms if tri_ms > 0 else float("inf")

    bytes_moved = 2 * args.rows * args.cols * x.element_size()
    pt_bw = bytes_moved / (pt_ms / 1000) / 1e9
    tri_bw = bytes_moved / (tri_ms / 1000) / 1e9

    print("\n=== Benchmark Result ===")
    print(f"PyTorch softmax:   {pt_ms:.4f} ms")
    print(f"Triton softmax:    {tri_ms:.4f} ms")
    print(f"Speedup:           {speedup:.2f}x")
    print(f"PyTorch bandwidth: {pt_bw:.2f} GB/s")
    print(f"Triton bandwidth:  {tri_bw:.2f} GB/s")


if __name__ == "__main__":
    main()
