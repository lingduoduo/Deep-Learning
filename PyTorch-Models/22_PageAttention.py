import math

import torch


def attention_kernel(query, full_kv):
    """
    query: (num_heads, head_dim)
    full_kv: (num_heads, seq_len, head_dim * 2)
    """
    keys, values = torch.chunk(full_kv, chunks=2, dim=-1)

    scores = torch.einsum("hd,hsd->hs", query, keys) / math.sqrt(query.size(-1))
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("hs,hsd->hd", attn_weights, values)

    return output, attn_weights


class PagedAttention:
    def __init__(self, num_heads, head_dim, block_size=16, num_blocks=1024):
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Physical block pool, allocated on demand.
        self.physical_blocks = torch.zeros(
            num_blocks,
            num_heads,
            block_size,
            head_dim * 2,
        )
        self.free_blocks = list(range(num_blocks))

        # Block table per sequence.
        self.block_tables = {}
        self.block_fill_counts = torch.zeros(num_blocks, dtype=torch.long)

    def allocate_sequence(self, seq_id):
        """Allocate a block table for a new sequence."""
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence '{seq_id}' already exists")

        self.block_tables[seq_id] = []

    def append_tokens(self, seq_id, new_kv, num_tokens=None):
        """Append KV data for newly generated tokens."""
        if seq_id not in self.block_tables:
            raise KeyError(f"Unknown sequence '{seq_id}'")

        if num_tokens is None:
            num_tokens = new_kv.size(0)

        block_table = self.block_tables[seq_id]

        for i in range(num_tokens):
            # Check whether the current block is full.
            if len(block_table) == 0 or self._is_block_full(block_table[-1]):
                # Allocate a new physical block.
                new_block_id = self.free_blocks.pop()
                block_table.append(new_block_id)

            # Store KV data into the selected block.
            block_id = block_table[-1]
            offset = self._get_block_offset(block_id)
            self.physical_blocks[block_id, :, offset, :] = new_kv[i]
            self.block_fill_counts[block_id] += 1

    def attention_compute(self, seq_id, query):
        """Compute attention using the sequence's block table."""
        if seq_id not in self.block_tables:
            raise KeyError(f"Unknown sequence '{seq_id}'")

        block_table = self.block_tables[seq_id]

        # Gather all KV blocks for this sequence.
        kv_blocks = []
        for block_id in block_table:
            used_tokens = self.block_fill_counts[block_id].item()
            kv_blocks.append(self.physical_blocks[block_id, :, :used_tokens, :])

        # Concatenate into a full KV sequence.
        full_kv = torch.cat(kv_blocks, dim=1)
        return attention_kernel(query, full_kv)

    def sequence_length(self, seq_id):
        block_table = self.block_tables[seq_id]
        return sum(self.block_fill_counts[block_id].item() for block_id in block_table)

    def _is_block_full(self, block_id):
        return self.block_fill_counts[block_id].item() >= self.block_size

    def _get_block_offset(self, block_id):
        return self.block_fill_counts[block_id].item()


def generate_sampled_kv_data(
    num_tokens,
    num_heads,
    head_dim,
    focus_token=None,
    noise_scale=0.15,
):
    """
    Build synthetic KV data for a single sequence.

    One token is made more salient so the demo attention distribution is easy
    to inspect after running the script.
    """
    keys = torch.randn(num_tokens, num_heads, head_dim) * noise_scale
    values = torch.randn(num_tokens, num_heads, head_dim) * noise_scale

    if focus_token is None:
        focus_token = torch.randint(0, num_tokens, (1,)).item()

    signal = torch.randn(num_heads, head_dim)
    signal = signal / signal.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    keys[focus_token] += 3.0 * signal
    values[focus_token] += 2.0 * signal

    kv = torch.cat([keys, values], dim=-1)

    metadata = {
        "focus_token": focus_token,
        "num_tokens": num_tokens,
    }
    return kv, signal, metadata


def run_demo():
    torch.manual_seed(7)

    num_heads = 4
    head_dim = 8
    block_size = 4
    num_blocks = 12

    paged_attention = PagedAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=num_blocks,
    )

    seq_id = "demo-sequence"
    paged_attention.allocate_sequence(seq_id)

    sampled_kv, query_hint, metadata = generate_sampled_kv_data(
        num_tokens=10,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    print("=== Sampled KV Data ===")
    print(f"sampled_kv shape: {sampled_kv.shape}")
    print(f"focus_token: {metadata['focus_token']}")
    print(f"block_size: {block_size}")

    paged_attention.append_tokens(seq_id, sampled_kv[:6], num_tokens=6)
    paged_attention.append_tokens(seq_id, sampled_kv[6:], num_tokens=4)

    print("\n=== Block Allocation ===")
    print(f"block_table: {paged_attention.block_tables[seq_id]}")
    print(f"sequence_length: {paged_attention.sequence_length(seq_id)}")

    query = query_hint + 0.05 * torch.randn(num_heads, head_dim)
    output, attn_weights = paged_attention.attention_compute(seq_id, query)

    mean_attn = attn_weights.mean(dim=0)
    top_token = mean_attn.argmax().item()

    print("\n=== Attention Demo ===")
    print(f"query shape: {query.shape}")
    print(f"output shape: {output.shape}")
    print(f"attention shape: {attn_weights.shape}")
    print(f"predicted_top_token: {top_token}")
    print(f"mean_attention: {[round(x, 3) for x in mean_attn.tolist()]}")


if __name__ == "__main__":
    run_demo()
