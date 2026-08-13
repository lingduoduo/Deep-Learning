import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import math

class LinearRegression:
    # Method 1 — Closed-Form (Normal Equation)
    def closed_form(self, X: torch.Tensor, y: torch.Tensor):
        n, d = X.shape
        # Augment X with ones column for bias
        X_aug = torch.cat([X, torch.ones(n, 1)], dim=1)  # (n, d+1)
        # Solve (X^T X) theta = X^T y
        theta = torch.linalg.lstsq(X_aug, y).solution      # (d+1,)
        w = theta[:d]
        b = theta[d]
        return w.detach(), b.detach()
    
    # Method 2 — Gradient Descent from Scratch
    def gradient_descent(self, X: torch.Tensor, y: torch.Tensor, 
                         lr: float = 0.01, steps: int = 1000):
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
    
    # Method 3 — PyTorch nn.Linear
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


# Simple Linear Regression Model
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


# Linear Regression
class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # a trainable model parameter
        self.w = nn.Parameter(
            torch.randn(in_features, out_features)
        )
        self.b = nn.Parameter(
            torch.zeros(out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w + self.b
# Create synthetic data
n = 200
in_features = 4
out_features = 2
X = torch.randn(n, in_features)
true_w = torch.tensor([
    [2.0, -1.0],
    [0.5,  3.0],
    [-2.0, 1.5],
    [1.0,  0.5]
])
true_b = torch.tensor([0.5, -1.0])
y = X @ true_w + true_b
# Create model
model = LinearRegression(
    in_features=in_features,
    out_features=out_features
)
# Loss + optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05
)
# Training loop
for step in range(1000):
    # 1. Clear old gradients
    optimizer.zero_grad()
    # 2. Forward pass
    pred = model(X)
    # 3. Calculate loss
    loss = loss_fn(pred, y)
    # 4. Calculate gradients
    loss.backward()
    # 5. Update parameters
    optimizer.step()
    if step % 100 == 0:
        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.6f}"
        )
# Check learned parameters
print("\nTrue W:")
print(true_w)
print("\nLearned W:")
print(model.w.detach())
print("\nTrue b:")
print(true_b)
print("\nLearned b:")
print(model.b.detach())


class MultipleLinearRegressionData(Dataset):
    def __init__(self, file):
        df = pd.read_csv(file)

        # x1, x2, x3
        self.X = torch.tensor(
            df.iloc[:, :-1].values,
            dtype=torch.float32
        )

        # y
        self.y = torch.tensor(
            df.iloc[:, -1].values,
            dtype=torch.float32
        ).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

torch.manual_seed(42)
X = torch.rand(100, 3) * 10
# y = 2*x1 + 3*x2 - x3 + 4 + noise
noise = torch.randn(100, 1) * 0.5
y = (
    2.0 * X[:, 0:1]
    + 3.0 * X[:, 1:2]
    - 1.0 * X[:, 2:3]
    + 4.0
    + noise
)
print("X shape:", X.shape)  # (100, 3)
print("y shape:", y.shape)  # (100, 1)

data = torch.cat((X, y), dim=1)
df = pd.DataFrame(
    data.numpy(),
    columns=["x1", "x2", "x3", "y"]
)
df.to_csv("data2.csv", index=False)


dataset = MultipleLinearRegressionData("data2.csv")
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
model = LinearRegression(
    in_features=3,
    out_features=1
)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)

epochs = 1000

for epoch in range(epochs):
    total_loss = 0.0

    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()

        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)

    average_loss = total_loss / len(dataset)

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {average_loss:.4f}"
        )

print("Learned weights:", model.w.detach().squeeze())
print("Learned bias:", model.b.detach())