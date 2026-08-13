import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_outputs),
        )

    def forward(self, x):
        return self.net(x)

# Generate data
X = torch.rand(100, 2) * 10
y = (
    X[:, 0] + 2 * X[:, 1]
).unsqueeze(1) + torch.randn(100, 1)
# Create model
model = MLP(
    num_inputs=2,
    num_hiddens=32,
    num_outputs=1
)
criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)
# Training
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )
# Test
X_test = torch.tensor([
    [4.0, 3.0],
    [7.0, 8.0]
])
with torch.no_grad():
    predictions = model(X_test)
    print("Predictions:", predictions)


# SwiGLU uses a gating mechanism: the gate projection controls information flow, while the up projection provides the content. This consistently outperforms standard FFNs in practice (PaLM, LLaMA, Mistral all use it).
class SwiGLUMLP(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super().__init__()

        # Both branches: input → hidden
        self.gate_proj = nn.Linear(num_inputs, num_hiddens)
        self.up_proj = nn.Linear(num_inputs, num_hiddens)

        # hidden → output
        self.down_proj = nn.Linear(num_hiddens, num_outputs)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up       # element-wise multiplication
        return self.down_proj(hidden)
    
X = torch.rand(100, 2) * 10
y = (
    X[:, 0] + 2 * X[:, 1]
).unsqueeze(1) + torch.randn(100, 1)
model = SwiGLUMLP(
    num_inputs=2,
    num_hiddens=32,
    num_outputs=1
)

criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )
X_test = torch.tensor([
    [4.0, 3.0],
    [7.0, 8.0]
])
with torch.no_grad():
    predictions = model(X_test)
print("Predictions:")
print(predictions)
