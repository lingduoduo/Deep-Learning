import torch
import torch.nn as nn
import torch.optim as optim


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x > 0,
        x,
        0.0
    )
x = torch.tensor([-2., -1., 0., 1., 2.])
print("Input: ", x)
print("Output:", relu(x))
print("Shape: ", relu(x).shape)


def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
x = torch.tensor([1.0, 2.0, 3.0])
print("Output:", my_softmax(x, dim=-1))
print("Sum:   ", my_softmax(x, dim=-1).sum())  # should be ~1.0
print("Ref:   ", torch.softmax(x, dim=-1))


def cross_entropy_loss(logits, targets):
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    batch_idx = torch.arange(logits.size(0))
    loss = - log_probs[batch_idx, targets]
    return loss.mean()
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
print('Loss:', cross_entropy_loss(logits, targets).item())
print('Ref: ', torch.nn.functional.cross_entropy(logits, targets).item())


class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.rand_like(x) > self.p
        return x * mask / (1 - self.p)
d = MyDropout(p=0.5)
d.train()
x = torch.ones(10)
print('Train:', d(x))
d.eval()
print('Eval: ', d(x))


class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, indices):
        return self.weight[indices]
emb = MyEmbedding(10, 4)
idx = torch.tensor([0, 3, 7])
print('Output shape:', emb(idx).shape)
print('Matches manual:', torch.equal(emb(idx)[0], emb.weight[0]))

