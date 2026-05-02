import numpy as np

import torch
import torch.nn.functional as F

# Top-k 采样：
# 做法：只保留概率最高的 k 个词，把剩下的词概率强制设为 0，然后重新归一化（让剩下的概率和为 1），再从中采样。
# 作用：直接砍掉长尾的低概率词，防止生成生僻字或乱码。
# 缺点：k 是固定的。如果模型很自信（某个词概率 90%），k 太大也会采样到噪音；如果模型很犹豫（概率很平），k 太小会限制多样性。

def top_k_filtering(logits, top_k=50, temperature=1.0, filter_value=-float('Inf')):
    logits = logits / temperature

    squeeze = False
    if logits.ndim == 1:
        logits = logits[None, :]
        squeeze = True

    _, vocab_size = logits.shape

    if top_k > 0 and top_k < vocab_size:
        kth_values = np.partition(logits, -top_k, axis=-1)[:, -top_k]
        indices_to_remove = logits < kth_values[:, None]
        logits[indices_to_remove] = filter_value

    if squeeze:
        return logits[0]

    return logits

logits = np.array([2.0, 1.0, 0.5, 0.1])
print(top_k_filtering(logits, top_k=2))
# output: [2.0, 1.0, -inf, -inf]


def top_k_filtering(logits, top_k=50, temperature=1.0, filter_value=-float('Inf')):
    logits = logits / temperature

    idx_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[idx_to_remove] = filter_value
    return logits
logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]]) 
print(top_k_filtering(logits, top_k=2))
# output: tensor([[2., 1., -inf, -inf]])

# Top-p (Nucleus) 采样：
# 做法：将词按概率从大到小排序，依次累加概率，直到累加和超过 p (比如 0.9)。保留这些词，剩下的截断，重新归一化，再采样。
# 作用：动态调整候选词数量。模型自信时候选词少，模型犹豫时候选词多。
# 现状：目前 LLM 推理中，Top-p 比 Top-k 更常用，或者两者结合。

def sortmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def top_p_filtering(logits, top_p=0.9, temperature=1.0, filter_value=-float("inf"), min_tokens_to_keep=1):

    logits = logits / temperature
    squeeze = False
    if logits.ndim == 1:
        logits = logits[None, :]  # add batch dim
        squeeze = True
    
    sorted_idx = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, sorted_idx, axis=-1)

    probs = sortmax(sorted_logits, axis=-1)
    cum_probs = np.cumsum(probs, axis=-1)

    sorted_indices_to_remove = cum_probs > top_p
    # shift right (HF trick)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = np.zeros_like(sorted_indices_to_remove, dtype=bool)
    np.put_along_axis(indices_to_remove, sorted_idx, sorted_indices_to_remove, axis=-1)

    logits[indices_to_remove] = filter_value
    return logits[0] if squeeze else logits

logits = np.array([2.0, 1.0, 0.1, 0.1, 0.1])
filtered_logits = top_p_filtering(logits, top_p=0.8, temperature=1.0)
print(filtered_logits)

def top_p_filtering(logits, top_p=0.9, temperature=1.0, filter_value=-float("inf"), min_tokens_to_keep=1):
    logits = logits / temperature

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    remove_sorted = cumulative_probs > top_p
    # shift right (HF trick)
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    remove_sorted[..., :min_tokens_to_keep] = False

    remove = torch.zeros_like(remove_sorted, dtype=torch.bool)
    remove.scatter_(dim=-1, index=sorted_idx, src=remove_sorted)

    logits[remove] = filter_value
    return logits[0] if squeeze else logits

logits = torch.tensor([[2.0, 1.0, 0.1, 0.1, 0.1]]) 
filtered_logits = top_p_filtering(logits, top_p=0.8, temperature=1.0)
print(filtered_logits)
    
