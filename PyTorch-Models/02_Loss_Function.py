import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


## Softmax Probabilities
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted_logits = logits - logits.amax(dim=dim, keepdim=True)
    exp_logits = shifted_logits.exp()
    return exp_logits / exp_logits.sum(dim=dim, keepdim=True)
x = torch.tensor([1.0, 2.0, 3.0])
logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]])
probs = softmax(logits)
print(f"Softmax Probabilities:\n{probs}")


# 手写交叉熵 Loss
# logits: [batch, vocab]
# targets: [batch] 类别索引
def cross_entropy_loss(logits, targets):
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    batch_idx = torch.arange(logits.size(0))
    loss = - log_probs[batch_idx, targets]
    return loss.mean()
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
print('Loss:', cross_entropy_loss(logits, targets).item())
print('Ref: ', torch.nn.functional.cross_entropy(logits, targets).item())


# HuberLoss
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

X = torch.rand(100, 1) * 10
y = 2 * X + 3 + torch.rand(100, 1)
pred = y + torch.randn_like(y) * 0.5
criterion = HuberLoss(delta=1.0)
loss = criterion(pred, y)
print(f"Huber Loss:{loss.item():.6f}")


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

