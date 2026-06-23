import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 计算 SFT 的交叉熵损失，自动处理 shift right 和忽略 padding。
# 参数:
#     logits: 模型输出的 logits，形状 (batch_size, seq_len, vocab_size)
#     labels: 真实 token ids，形状 (batch_size, seq_len)，其中填充部分设为 ignore_index
# 返回: 标量损失

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


# 手写交叉熵 Loss
# logits: [batch, vocab]
# targets: [batch] 类别索引
def softmax(logits, dim=-1):
    shifted_logits = logits - logits.amax(dim=dim, keepdim=True)
    exp_logits = shifted_logits.exp()
    return exp_logits / exp_logits.sum(dim=dim, keepdim=True)

logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]])
probs = softmax(logits)
print(f"Softmax Probabilities:\n{probs}")

def cross_entropy_loss(logits, targets):
    if logits.dim() != 2:
        raise ValueError(f"Expected logits with shape [batch, vocab], got {tuple(logits.shape)}")
    if targets.dim() != 1:
        raise ValueError(f"Expected targets with shape [batch], got {tuple(targets.shape)}")
    if logits.size(0) != targets.size(0):
        raise ValueError("Batch size of logits and targets must match")
    if torch.any(targets < 0) or torch.any(targets >= logits.size(1)):
        raise ValueError("Targets must contain valid class indices")

    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -target_log_probs.mean()

logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]])
targets = torch.tensor([0, 1]) 
loss = cross_entropy_loss(logits, targets)
print(f"Cross Entropy Loss:{loss.item():.6f}")


# HuberLoss
X = torch.rand(100, 1) * 10
y = 2 * X + 3 + torch.rand(100, 1)
class HuberLoss(nn.Module):
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, yhat, y):
        abs_error = torch.abs(y - yhat)

        quadratic = torch.minimum(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic

        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean()


# Contrastive Loss
# Implement CLIP Style contrastive loss, given a batch of images and texts 
# loss_i = cross_entropy(similarity_matrix, labels) 
# loss_t = cross_entropy(similarity_matrix.T, labels) 
# loss = (loss_i + loss_t) / 2

class CLIPContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        image_embeds: (batch_size, dim)
        text_embeds:  (batch_size, dim)

        Assumption:
        image_embeds[i] matches text_embeds[i]
        """

        # Normalize so dot product becomes cosine similarity
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Similarity matrix: (batch_size, batch_size)
        similarity_matrix = image_embeds @ text_embeds.T

        # Scale by temperature
        logits = similarity_matrix / self.temperature

        batch_size = image_embeds.size(0)
        labels = torch.arange(batch_size, device=image_embeds.device)

        # Image -> Text loss
        loss_i = F.cross_entropy(logits, labels)

        # Text -> Image loss
        loss_t = F.cross_entropy(logits.T, labels)

        loss = (loss_i + loss_t) / 2
        return loss

image_embeds = torch.randn(4, 512)
text_embeds = torch.randn(4, 512)

criterion = CLIPContrastiveLoss(temperature=0.07)
loss = criterion(image_embeds, text_embeds)
print(loss)
