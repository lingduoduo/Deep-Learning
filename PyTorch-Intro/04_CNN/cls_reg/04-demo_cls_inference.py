import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import os
# from CNN import CNN

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.con = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = torch.nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        out = self.con(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# --- paths ---
dir_path = os.path.dirname(os.path.abspath(__file__))     # .../04/cls_reg
parent_dir = os.path.dirname(dir_path)                    # .../04
ckpt_path = os.path.join(dir_path, "model", "mnist_model.pth")  # saved by training

# --- data (set download=True if first time) ---
test_data = dataset.MNIST(
    root="mnist",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=False)

# --- model load (state_dict) ---
cnn = CNN()
state = torch.load(ckpt_path, map_location="cpu")
cnn.load_state_dict(state)
cnn.eval()

# --- evaluation ---
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = cnn(images)
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test accuracy: {accuracy:.4f}")
