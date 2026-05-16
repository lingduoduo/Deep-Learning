import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import math


class LinearRegression:
    def closed_form(self, X: torch.Tensor, y: torch.Tensor):
        n, d = X.shape
        # Augment X with ones column for bias
        X_aug = torch.cat([X, torch.ones(n, 1)], dim=1)  # (n, d+1)
        # Solve (X^T X) theta = X^T y
        theta = torch.linalg.lstsq(X_aug, y).solution      # (d+1,)
        w = theta[:d]
        b = theta[d]
        return w.detach(), b.detach()


    def gradient_descent(self, X: torch.Tensor, y: torch.Tensor, lr: float = 0.01, steps: int = 1000):
        n, d = X.shape
        w = torch.zeros(d)
        b = torch.tensor(0.0)

        for _ in range(steps):
            pred = X @ w + b          # (n,)
            error = pred - y           # (n,)
            grad_w = (2.0 / n) * (X.T @ error)  # (d,)
            grad_b = (2.0 / n) * error.sum()     # scalar
            w = w - lr * grad_w
            b = b - lr * grad_b
        return w, b

    def nn_linear(self, X: torch.Tensor, y: torch.Tensor,
                  lr: float = 0.01, steps: int = 1000):
        """PyTorch nn.Linear with autograd training loop."""
        n, d = X.shape
        model = nn.Linear(d, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(steps):
            optimizer.zero_grad()
            pred = model(X).squeeze(-1)  # (n,)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        w = model.weight.data.squeeze(0)  # (d,)
        b = model.bias.data.squeeze(0)    # scalar ()
        return w, b

# Verify
torch.manual_seed(42)
X = torch.randn(100, 3)
true_w = torch.tensor([2.0, -1.0, 0.5])
y = X @ true_w + 3.0

model = LinearRegression()
for name, method in [("Closed-form", model.closed_form),
                      ("Grad Descent", lambda X, y: model.gradient_descent(X, y, lr=0.05, steps=2000)),
                      ("nn.Linear", lambda X, y: model.nn_linear(X, y, lr=0.05, steps=2000))]:
    w, b = method(X, y)
    print(f"{name:13s}  w={w.tolist()}  b={b.item():.4f}")
print(f"{'True':13s}  w={true_w.tolist()}  b=3.0000")


# Load Data
class LinearRegressionData(Dataset):
    def __init__(self, file):
        df = pd.read_csv(file)
        self.X = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X = torch.rand(100, 1) * 10
y = 2 * X + torch.rand(100, 1)
data = torch.cat((X, y), dim=1)
df = pd.DataFrame(data.numpy(), columns=['X', 'y'])
df.to_csv('data.csv', index = False)

dataset = LinearRegressionData("data.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Simple Linear Model
class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.w = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
        self.w.requires_grad_(True)
        self.b = torch.zeros(out_features, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.T + self.b
    
model = SimpleLinear(8, 4)
print("W shape:", model.w.shape)
print("b shape:", model.b.shape)
x = torch.randn(2, 8)
print("Output shape:", model.forward(x).shape)


# Simple Linear Model
class LinearRegressionData(Dataset):
    def __init__(self, file):
        df = pd.read_csv(file)
        self.X = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        prediction = model(batch_X)
        loss = criterion(prediction, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

[w, b] = model.linear.parameters()
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

X_test = torch.tensor([[4.0], [7.0]])
with torch.no_grad():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
