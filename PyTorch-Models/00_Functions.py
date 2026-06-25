import torch
import torch.nn as nn
import torch.optim as optim
import math


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


# GELU (Gaussian Error Linear Unit) activation
def my_gelu(x):
    return 0.5 * x * ( 1 + torch.erf(x / math.sqrt(2.0)))
x = torch.tensor([-2., -1., 0., 1., 2.])
print('Output:', my_gelu(x))
print('Ref:   ', torch.nn.functional.gelu(x))


# Kaiming Initialization
def kaiming_init(weight):
    fan_in = weight.shape[1] if weight.dim() < 2 else weight.shape[0]
    std = math.sqrt(2/fan_in)
    with torch.no_grad():
        weight.normal_(0, std)
    return weight
w = torch.empty(256, 512)
kaiming_init(w)
print(f'Mean: {w.mean():.4f} (expect ~0)')
print(f'Std:  {w.std():.4f} (expect {math.sqrt(2/512):.4f})')


# Gradient Norm Clipping
def clip_grad_norm(parameters, max_norm):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in parameters))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.mul_(clip_coef)
    return total_norm.item()
p = torch.randn(100, requires_grad=True)
(p * 10).sum().backward()
print('Before:', p.grad.norm().item())
orig = clip_grad_norm([p], max_norm=1.0)
print('After: ', p.grad.norm().item())
print('Original norm:', orig)


Gradient Accumulation
def accumulated_step(model, optimizer, loss_fn, micro_batches):
    optimizer.zero_grad()
    total_loss = 0.0
    n = len(micro_batches)
    for x, y in micro_batches:
        loss = loss_fn(model(x), y) / n
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return loss
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss = accumulated_step(model, opt, nn.MSELoss(),
    [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(4)])
print('Loss:', loss)


