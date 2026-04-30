import torch
import torch.nn.functional as F

# Top-k 采样：
# 做法：只保留概率最高的 k 个词，把剩下的词概率强制设为 0，然后重新归一化（让剩下的概率和为 1），再从中采样。
# 作用：直接砍掉长尾的低概率词，防止生成生僻字或乱码。
# 缺点：k 是固定的。如果模型很自信（某个词概率 90%），k 太大也会采样到噪音；如果模型很犹豫（概率很平），k 太小会限制多样性。

def top_k_filtering(logits, top_k=50, temperature=1.0, filter_value=-float('Inf')):
    logits = logits / temperature

    idx_to_remove = logits < torch.top_k(logits, top_k)[0][..., -1, None]
    logits[idx_to_remove] = filter_value
    return logits

# Top-p (Nucleus) 采样：
# 做法：将词按概率从大到小排序，依次累加概率，直到累加和超过 p (比如 0.9)。保留这些词，剩下的截断，重新归一化，再采样。
# 作用：动态调整候选词数量。模型自信时候选词少，模型犹豫时候选词多。
# 现状：目前 LLM 推理中，Top-p 比 Top-k 更常用，或者两者结合。

def top_p_filtering(logits, top_p=0.9, temperature=1.0, filter_value=-float("inf"), min_tokens_to_keep=1):
    logits = logits / temperature

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_idx, sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits

if __name__ == "__main__":
    logits = torch.tensor([[2.0, 1.0, 0.1, 0.1, 0.1]]) 
    print("Before: ", logits)
    
    filtered_logits = top_p_filtering(logits, top_p=0.8, temperature=1.0)
    print("After:", filtered_logits)
    
