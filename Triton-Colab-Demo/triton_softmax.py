import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"num_warps": nw}) for nw in [1, 2, 4, 8, 16, 32]],
    key=["n_cols"],
)
@triton.jit
def _softmax_kernel(
    x_ptr,
    y_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    row_start = row_id * n_cols
    mask = offsets < n_cols

    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float("inf"))

    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    tl.store(y_ptr + row_start + offsets, y, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    x: [rows, cols], CUDA tensor.
    Returns row-wise softmax.
    """
    if not x.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if x.dim() != 2:
        raise ValueError("Input must be 2D: [rows, cols].")

    x = x.contiguous()
    rows, cols = x.shape
    block_size = triton.next_power_of_2(cols)

    if block_size > 131072:
        raise ValueError("cols exceeds maximum supported block size of 131072.")

    y = torch.empty_like(x)

    _softmax_kernel[(rows,)](x, y, cols, BLOCK_SIZE=block_size)

    return y
