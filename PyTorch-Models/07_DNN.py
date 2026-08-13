import torch
import torch.nn as nn
import torch.optim as optim



class DNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)
    
X = torch.rand(100, 2) * 10
y = (X[:, 0] + X[:, 1] * 2).unsqueeze(1) + torch.randn(100, 1) 

model = DNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
input_dim = 2
hidden_dim = 32
output_dim = 1

for epoch in range(epochs):
    optimizer.zero_grad()

    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

X_test = torch.tensor([[4.0, 3.0], [7.0, 8.0]])
with torch.no_grad():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

for param in model.parameters():
    print(param)