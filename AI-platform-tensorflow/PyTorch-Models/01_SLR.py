import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd

X = torch.rand(100, 1) * 10
y = 2 * X + torch.rand(100, 1)
data = torch.cat((X, y), dim=1)
df = pd.DataFrame(data.numpy(), columns=['X', 'y'])
df.to_csv('data.csv', index = False)

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

dataset = LinearRegressionData("data.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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


X_test = torch.tensor([4.0], [7.0])
with torch.no_grad():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
