import torch
import torch.nn as nn

# 计算 SFT 的交叉熵损失，自动处理 shift right 和忽略 padding。
# 参数:
#     logits: 模型输出的 logits，形状 (batch_size, seq_len, vocab_size)
#     labels: 真实 token ids，形状 (batch_size, seq_len)，其中填充部分设为 ignore_index
# 返回:
#     标量损失

def sft_loss(logits, labels, ignore_index=-100):
    if logits.dim() != 3:
        raise ValueError(f"Expected logits with shape (batch, seq_len, vocab_size), got {tuple(logits.shape)}")
    if labels.dim() != 2:
        raise ValueError(f"Expected labels with shape (batch, seq_len), got {tuple(labels.shape)}")

    shift_logits = logits[:, :-1, :].contiguous() # (batch, seq_len-1, vocab)
    shift_labels = labels[:, 1:].contiguous() # (batch, seq_len-1)

    loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

batch_size, seq_len, vocab_size = 2, 5, 10
logits = torch.randn(batch_size, seq_len, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))

labels[:, 3:] = -100

loss = sft_loss(logits, labels)
print("SFT Loss:", loss.item())
