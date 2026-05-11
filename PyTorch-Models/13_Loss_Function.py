import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# HuberLoss
X = torch.rand(100, 1) * 10
y = 2 * X + 3 + torch.rand(100, 1)

class HuberLoss(nn.Module):
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, yhat, y):
        abs_error = torch.abs(y - yhat)

        quadratic = torch.minimum(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic

        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input and single output

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
criterion = HuberLoss(delta=1.0)
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()

    prediction = model(X)
    loss = criterion(prediction, y)
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


# Contrastive Loss
# Implement CLIP Style contrastive loss, given a batch of images and texts 
# loss_i = cross_entropy(similarity_matrix, labels) 
# loss_t = cross_entropy(similarity_matrix.T, labels) 
# loss = (loss_i + loss_t) / 2

class CLIPContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        image_embeds: (batch_size, dim)
        text_embeds:  (batch_size, dim)

        Assumption:
        image_embeds[i] matches text_embeds[i]
        """

        # Normalize so dot product becomes cosine similarity
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Similarity matrix: (batch_size, batch_size)
        similarity_matrix = image_embeds @ text_embeds.T

        # Scale by temperature
        logits = similarity_matrix / self.temperature

        batch_size = image_embeds.size(0)
        labels = torch.arange(batch_size, device=image_embeds.device)

        # Image -> Text loss
        loss_i = F.cross_entropy(logits, labels)

        # Text -> Image loss
        loss_t = F.cross_entropy(logits.T, labels)

        loss = (loss_i + loss_t) / 2
        return loss

image_embeds = torch.randn(4, 512)
text_embeds = torch.randn(4, 512)

criterion = CLIPContrastiveLoss(temperature=0.07)
loss = criterion(image_embeds, text_embeds)
print(loss)
