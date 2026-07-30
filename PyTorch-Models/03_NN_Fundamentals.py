import torch
import torch.nn as nn
import math

# 1. ReLu
def relu(x: torch.Tensor) -> torch.Tensor:
    # return x * (x > 0).float()
    return torch.where(x >= 0, x, 0)

x = torch.tensor([-2., -1., 0., 1., 2.])
print("Input: ", x)
print("Output:", relu(x))
print("Shape: ", relu(x).shape)


# 2. SoftMax
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

x = torch.tensor([1.0, 2.0, 3.0])
print("Output:", softmax(x, dim=-1))
print("Sum:   ", softmax(x, dim=-1).sum())  # should be ~1.0
print("Ref:   ", torch.softmax(x, dim=-1))


# 3. Cross Entropy Loss
def cross_entropy_loss(logits, targets):
    # torch.logsumexp = torch.log(torch.sum(torch.exp(logits)))
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # -log_probs[torch.arange(targets.shape[0]), targets].mean()
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1), ).squeeze(-1)
    return -target_log_probs.mean()

logits = torch.randn(4, 10)
print(logits)
targets = torch.randint(0, 10, (4,))
print(targets)
print('Loss:', cross_entropy_loss(logits, targets).item())
print('Ref: ', torch.nn.functional.cross_entropy(logits, targets).item())


# 4. Dropout
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
        self.p = p

    def forward(self, x):        
        if not self.training or self.p == 0:
            return x

        if self.p == 1:
            return torch.zeros_like(x)

        keep_prob = 1 - self.p
        mask = torch.rand_like(x) < keep_prob

        return x * mask / keep_prob

d = Dropout(p=0.5)
d.train()
x = torch.ones(10)
print('Train:', d(x))
d.eval()
print('Eval: ', d(x))


# 5. Embedding Layer
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, indices):
        return self.weight[indices]

emb = Embedding(10, 4)
idx = torch.tensor([0, 3, 7])
print(emb.weight[idx])
print('Output shape:', emb(idx).shape)
print('Matches manual:', torch.equal(emb(idx)[0], emb.weight[0]))


# 6. Gaussian Error Linear Unit
def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
x = torch.tensor([-2., -1., 0., 1., 2.])
print('Output:', gelu(x))
print('Ref:   ', torch.nn.functional.gelu(x))


# 7. Kaiming (He) normal initialization
def kaiming_init(weight):
    if weight.dim() < 2:
        raise ValueError(
            "Kaiming initialization requires at least 2 dimensions"
        )
    fan_in = weight.shape[1]
    if weight.dim() > 2:
        receptive_field_size = math.prod(weight.shape[2:])
        fan_in *= receptive_field_size
    std = math.sqrt(2.0 / fan_in)
    with torch.no_grad():
        weight.normal_(mean=0.0, std=std)
    return weight

w = torch.empty(256, 512)
kaiming_init(w)
print(f'Mean: {w.mean():.4f} (expect ~ 0)')
print(f'Std:  {w.std():.4f} (expect {math.sqrt(2/512):.4f})')


# 8. Gradient Norm Clipping
def clip_grad_norm(parameters, max_norm):
    parameters = [
        p for p in parameters
        if p.grad is not None
    ]

    if not parameters:
        return 0.0

    grad_norms = torch.stack([
        p.grad.detach().norm(2)
        for p in parameters
    ])

    total_norm = grad_norms.norm(2)

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)

    for p in parameters:
        p.grad.mul_(clip_coef)

    return total_norm.item()

p = torch.randn(100, requires_grad=True)
(p * 10).sum().backward()
print('Before:', p.grad.norm().item())
orig = clip_grad_norm([p], max_norm=1.0)
print('After: ', p.grad.norm().item())
print('Original norm:', orig)
     

# 9. Gradient Accumulation
def accumulated_step(model, optimizer, loss_fn, micro_batches):
    optimizer.zero_grad()
    n = len(micro_batches)
    total_loss = 0.0
    for x, y in micro_batches:
        loss = loss_fn(model(x), y) / n
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss

model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss = accumulated_step(model, opt, nn.MSELoss(),
    [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(4)])
print('Loss:', loss)


def accumulated_step_unequal(model, optimizer, loss_fn, micro_batches):
    optimizer.zero_grad()

    total_examples = sum(x.shape[0] for x, _ in micro_batches)
    total_loss = 0.0

    for x, y in micro_batches:
        loss = loss_fn(model(x), y)
        batch_size = x.shape[0]
        weight = batch_size / total_examples

        (loss * weight).backward()
        total_loss += loss.item() * batch_size

    optimizer.step()

    return total_loss / total_examples

micro_batches = [
    (torch.randn(2, 4), torch.randn(2, 2)),
    (torch.randn(5, 4), torch.randn(5, 2)),
    (torch.randn(1, 4), torch.randn(1, 2)),
]

loss = accumulated_step_unequal(
    model,
    opt,
    nn.MSELoss(),
    micro_batches,
)

print("Loss:", loss)