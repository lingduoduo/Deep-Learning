import torch
import torch.nn as nn
import torch.optim as optim

X = torch.rand(100, 1) * 10
y = 2 * X + 3 + torch.rand(100, 1)

class CustomActivationModel(nn.Module):
    def __init__(self):
        super(CustomActivationModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        x = self.linear(x)
        return x + torch.tanh(x)

model = CustomActivationModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()

    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


[w, b] = model.linear.parameters()
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

# Testing on new data
X_test = torch.tensor([[4.0], [7.0]])
with torch.no_grad():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

