import torch
import torch.nn as nn

# 1. ReLu
def relu(x: torch.Tensor) -> torch.Tensor:
    # return x * (x > 0).float()
    return torch.where(x >= 0, x,0)

x = torch.tensor([-2., -1., 0., 1., 2.])
print("Input: ", x)
print("Output:", relu(x))
print("Shape: ", relu(x).shape)

# 2. SoftMax
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_scaled = torch.exp(x - x_max)
    return torch.exp(x_scaled) / torch.sum(torch.exp(x_scaled))

x = torch.tensor([1.0, 2.0, 3.0])
print("Output:", softmax(x, dim=-1))
print("Sum:   ", softmax(x, dim=-1).sum())  # should be ~1.0
print("Ref:   ", torch.softmax(x, dim=-1))

# 3. Cross Entropy Loss
def cross_entropy_loss(logits, targets):
    # torch.logsumexp = torch.log(torch.sum(torch.exp(logits)))
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -log_probs[torch.arange(targets.shape[0]), targets].mean()

logits = torch.randn(4, 10)
print(logits)
targets = torch.randint(0, 10, (4,))
print(targets)
print('Loss:', cross_entropy_loss(logits, targets).item())
print('Ref: ', torch.nn.functional.cross_entropy(logits, targets).item())

# 4. Dropout
class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 1:
            return x
        
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1 - self.p)

d = MyDropout(p=0.5)
d.train()
x = torch.ones(10)
print('Train:', d(x))
d.eval()
print('Eval: ', d(x))

# 5. Embedding Layer
class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, indices):
        return self.weight[indices]

emb = MyEmbedding(10, 4)
idx = torch.tensor([0, 3, 7])
print(emb.weight[idx])
print('Output shape:', emb(idx).shape)
print('Matches manual:', torch.equal(emb(idx)[0], emb.weight[0]))

