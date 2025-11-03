# Data Loader
'''
Batch loading: It can split the dataset into multiple smaller batches, which is a standard practice in deep learning training that improves both training efficiency and memory utilization.

Data shuffling: By setting shuffle=True, the dataset order is randomized before each training epoch begins. This adds randomness to the training process and improves model generalization.

Multi-threaded loading: The num_workers parameter enables multi-threaded data loading in the background, reducing I/O waiting time and accelerating training.

Memory saving: DataLoader loads data on demand rather than loading the entire dataset into memory at once. This is especially important for large-scale datasets.
'''
from torch.utils.data import DataLoader
from your_dataset_module import YourCustomDataset

custom_dataset = CustomDataset(data, labels, transform=...)
data_loader = DataLoader(
    dataset=custom_dataset,  # Dataset instance
    batch_size=32,           # Number of samples per batch
    shuffle=True,            # Shuffle data at the start of each epoch
    num_workers=4,           # Number of subprocesses for data loading (0 = no multithreading)
    drop_last=False,         # Drop the last incomplete batch if dataset size is not divisible by batch_size
)

# Using data_loader in a training loop
for inputs, labels in data_loader:
    # Perform model training or validation here
    pass


'''
In PyTorch, the torchvision.transforms module provides a variety of preprocessing and transformation functions. The following are some commonly used transformation operations:

1. Resize: Resize the image to a specific size. Example: transforms.Resize((256, 256)).
2. CenterCrop: Crop a central region of the image. Example: transforms.CenterCrop(224).
3. RandomCrop: Randomly crop a region to increase data diversity.
4. RandomHorizontalFlip: Randomly flip images horizontally as data augmentation.
5. RandomRotation: Randomly rotate the image for further augmentation.
6. ToTensor: Convert a PIL image or numpy array into a PyTorch tensor and rearrange color channels from HWC to CHW.
7. Normalize: Normalize pixel values using dataset-specific mean and std (e.g., ImageNet mean/std values).
'''
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomDataset(data, labels, transform=transform)

# Dataset definition
'''
In PyTorch, custom datasets are usually defined by creating a class that inherits from torch.utils.data.Dataset.
This class must implement two core methods:

__len__: Returns the number of samples in the dataset.
__getitem__: Returns one sample (and label if available) based on a given index. This method is typically responsible for loading and preprocessing data.
'''
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

'''
The CustomDataset class takes three parameters: data, labels, and transform, representing the dataset samples, labels, and preprocessing transformations.
__init__ stores these for later use. __len__ returns dataset size, and __getitem__ retrieves a specific data/label pair.
Transformations (e.g., scaling, cropping, rotation) can be applied for data augmentation.
'''

import torch
import torch.nn as nn

# First model
class CustomLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomLayer, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.output_linear(x)
        return x

input_size = 10
hidden_size = 20
output_size = 5
model = CustomLayer(input_size, hidden_size, output_size)
print(model)

# Second model
class SimpleLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = SimpleLayer(input_dim, hidden_dim)
        self.layer2 = SimpleLayer(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

input_dim = 784
hidden_dim = 128
output_dim = 10
model = MLP(input_dim, hidden_dim, output_dim)
print(model)

# Third model
class ComplexModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ComplexModel, self).__init__()
        self.custom_layer = CustomLayer(input_size, hidden_size, output_size)
        self.mlp = MLP(hidden_size, hidden_size, output_size)

    def forward(self, x):
        x = self.custom_layer(x)
        x = self.mlp(x)
        return x

input_size = 784
hidden_size = 128
output_size = 10
model = ComplexModel(input_size, hidden_size, output_size)
print(model)

# Access model parameters
model = ComplexModel(input_size, hidden_size, output_size)
for param in model.parameters():
    print(param)

for name, param in model.named_parameters():
    print(f"Name: {name}, Parameter: {param}")

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Custom normalization layer
class CustomNormalization(nn.Module):
    def __init__(self, dim):
        super(CustomNormalization, self).__init__()
        self.dim = dim

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        std = x.std(dim=self.dim, keepdim=True)
        return (x - mean) / (std + 1e-8)

model.add_module('custom_normalization', CustomNormalization(1))

# Optimizers
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)

# Loss Functions
outputs = model(inputs)
targets = labels

mse_loss = nn.MSELoss()
loss = mse_loss(outputs, targets)

outputs = model(inputs)
targets = torch.LongTensor(labels)
cross_entropy_loss = nn.CrossEntropyLoss()
loss = cross_entropy_loss(outputs, targets)

outputs = model(inputs)
targets = labels.float()
bce_loss = nn.BCELoss()
loss = bce_loss(outputs, targets)

num_samples = [100, 1000]
weights = [1 / num_samples[0], 1 / num_samples[1]]
weighted_bce_loss = nn.BCEWithLogitsLoss(weight=torch.tensor(weights))
outputs = model(inputs)
labels = labels.float()
loss = weighted_bce_loss(outputs, labels)

# KL Divergence
preds = torch.randn(3, 5).softmax(dim=1)
targets = torch.randn(3, 5).softmax(dim=1)
criterion = nn.KLDivLoss(reduction='batchmean')
loss = criterion(preds.log(), targets)
print('KLDivLoss:', loss.item())

# TripletMarginLoss
torch.manual_seed(42)
feature_dim = 128
num_triplets = 10
anchors = torch.randn(num_triplets, feature_dim)
positives = torch.randn(num_triplets, feature_dim)
negatives = torch.randn(num_triplets, feature_dim)
triplets = torch.stack((anchors, positives, negatives), dim=1)
triplet_loss = nn.TripletMarginLoss(margin=1.0)
loss = triplet_loss(triplets[:, 0], triplets[:, 1], triplets[:, 2])
print('Triplet Margin Loss:', loss.item())

# Training models
output = model(inputs)
loss = criterion(output, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Learning Rate Scheduler Examples
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR

model = YourModel()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
for epoch in range(num_epochs):
    scheduler.step()

scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
for epoch in range(num_epochs):
    scheduler.step()

scheduler = ExponentialLR(optimizer, gamma=0.9)
for epoch in range(num_epochs):
    scheduler.step()

# Learning Rate Scheduler Example
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

num_epochs = 30
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], LR: {scheduler.get_last_lr()[0]}")

# Model Evaluation
model.eval()
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)

# Accuracy, Precision, Recall, F1, AUC Metrics
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

_, predicted = torch.max(outputs.data, 1)
predicted = predicted.cpu().numpy()
labels = labels.cpu().numpy()
accuracy = (predicted == labels).sum() / len(labels)
print(f'Accuracy: {accuracy}')

precision_macro = precision_score(true_labels, predictions, average='macro')
recall_macro = recall_score(true_labels, predictions, average='macro')
f1_macro = f1_score(true_labels, predictions, average='macro')
print(f'Macro Precision: {precision_macro}, Recall: {recall_macro}, F1: {f1_macro}')

# ROC-AUC Example
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
y_scores = [0.1, 0.4, 0.35, 0.8, 0.6, 0.1, 0.9, 0.7, 0.2, 0.5]
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Save and Load Models
model = SimpleModel()
torch.save(model.state_dict(), 'model.pth')
model = ComplexModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('model.pth'))

loaded_model = torch.load('model.pth', map_location=torch.device('cpu'))
torch.save(model, 'model_full.pth')
loaded_model = torch.load('model_full.pth')
