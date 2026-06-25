import time
import torch


class BenchmarkMeter:
    def __init__(self):
        self.start_time = None
        self.tokens = 0

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.start_time = time.time()

    def update(self, input_ids):
        self.tokens += int(input_ids.numel())

    def summary(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        else:
            peak_gb = 0.0

        elapsed = time.time() - self.start_time
        tokens_per_sec = self.tokens / max(elapsed, 1e-9)
        return {
            "elapsed_sec": elapsed,
            "tokens": self.tokens,
            "tokens_per_sec": tokens_per_sec,
            "peak_gpu_memory_gb": peak_gb,
        }


def print_summary(name, metrics):
    print("\n" + "=" * 80)
    print(f"{name} Summary")
    print("=" * 80)
    print(f"Total time:          {metrics['elapsed_sec']:.2f} sec")
    print(f"Tokens processed:    {metrics['tokens']}")
    print(f"Tokens/sec:          {metrics['tokens_per_sec']:.2f}")
    print(f"Peak GPU memory:     {metrics['peak_gpu_memory_gb']:.2f} GB")
    print("=" * 80 + "\n")
