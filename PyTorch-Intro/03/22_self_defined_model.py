# Data Loader
'''
Batch loading: It can split the data set into multiple small batches (batch), which is a standard practice in the deep learning training process, helping to improve training efficiency and memory utilization.

Data shuffling: By setting shuffle=True, the order of the data set can be randomly shuffled before the start of each training epoch, increasing the randomness of model training and helping to improve the generalization ability of the model.

Multi-threaded loading: Through the num_workers parameter, multiple threads can be used to load data concurrently in the background, reducing data I/O waiting time and further accelerating the training process.

Memory saving: DataLoader avoids loading the entire data set into memory at once by loading data on demand (that is, loading data from disk to memory only when needed during training), which is especially important for large-scale data sets.
'''
from torch.utils.data import DataLoader
from your_dataset_module import YourCustomDataset

custom_dataset = CustomDataset(data, labels,transform=...)
data_loader = DataLoader(
    dataset=custom_dataset,  # 数据集实例
    batch_size=32,           # 每个批次的样本数
    shuffle=True,            # 是否在每个epoch开始时打乱数据
    num_workers=4,           # 使用的子进程数，用于数据加载（0表示不使用多线程）
    drop_last=False,         # 如果数据集大小不能被batch_size整除，是否丢弃最后一个不完整的batch
)

# 使用data_loader在训练循环中迭代获取数据
for inputs, labels in data_loader:
    # 在这里执行模型训练或验证的代码
    pass


'''
In PyTorch, the torchvision.transforms module provides rich preprocessing and transformation functions. The following are some commonly used transformation operations:
"Common preprocessing and conversion operations:"
Resize: Resize the image to the specified size. For example, transforms.Resize((256, 256)) will resize the image to 256x256 pixels.

CenterCrop: Crop an area of ​​specified size from the center of the image, such as transforms.CenterCrop(224).

RandomCrop: Randomly crops an area of ​​a specified size from the image, which increases data diversity and is conducive to model learning.

RandomHorizontalFlip: Flips images horizontally with a certain probability, which is one of the commonly used data enhancement methods.

RandomRotation: Randomly rotate the image to a certain angle to further enhance data diversity.

ToTensor: Convert a PIL image or numpy array to PyTorch's Tensor and adjust the color channels from RGB to the format expected by Tensorflow (HWC -> CHW).

Normalize: Normalize image pixel values, usually using the mean and standard deviation of a specific data set (such as ImageNet), such as transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

In PyTorch, the torchvision.transforms module provides rich preprocessing and transformation functions. The following are some commonly used transformation operations:
"Common preprocessing and conversion operations:"
1. Resize: Resize the image to the specified size. For example, transforms.Resize((256, 256)) will resize the image to 256x256 pixels.

2. CenterCrop: Crop an area of ​​specified size from the center of the image, such as transforms.CenterCrop(224).

3. RandomCrop: Randomly crops an area of ​​a specified size from the image, which increases data diversity and is conducive to model learning.

4. RandomHorizontalFlip: Flips images horizontally with a certain probability, which is one of the commonly used data enhancement methods.

5. RandomRotation: Randomly rotate the image to a certain angle to further enhance data diversity.

6. ToTensor: Convert a PIL image or numpy array to PyTorch's Tensor and adjust the color channels from RGB to the format expected by Tensorflow (HWC -> CHW).

7. Normalize: Normalize image pixel values, usually using the mean and standard deviation of a specific data set (such as ImageNet), such as transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
'''

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
In PyTorch, the definition of a dataset is usually achieved by creating a new class, which inherits from torch.utils.data.Dataset. This custom class needs to implement the following two core methods:
__len__ method: This method returns the number of samples in the dataset. It tells the outside caller how many samples in the data set can be used to train or test the model.
__getitem__ method: This method returns a sample data and its corresponding label (if any) based on the given index. It allows on-demand access to any sample in the data set, usually including data loading and necessary preprocessing.
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
Among them, the CustomDataset class receives three parameters: data, labels, and transform, which respectively represent the sample data in the data set and the corresponding labels and preprocessing transformations. 
In the __init__ method, store these parameters in attributes of the class for access in the __getitem__ method. 
The __len__ method returns the length of the data set, which is the number of samples. The __getitem__ method obtains the corresponding data and labels through the index index.
In addition, in order to enhance the diversity of data and the generalization ability of the model, you can integrate data preprocessing logic in the dataset class, or dynamically transform the data by passing a transformation object (such as the transformation in torchvision.transforms), Such as rotation, scaling, cropping, etc.
'''   



import torch
import torch.nn as nn

## Fist model
class CustomLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomLayer, self).__init__()

        # 创建线性层
        self.linear = nn.Linear(input_size, hidden_size)

        # 创建ReLU激活函数
        self.relu = nn.ReLU()

        # 创建输出线性层（如果需要的话，例如对于分类任务）
        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 应用线性变换
        x = self.linear(x)

        # 应用ReLU激活函数
        x = self.relu(x)

        # 如果需要，可以添加更多的操作，例如另一个线性层
        x = self.output_linear(x)

        return x

input_size = 10
hidden_size = 20
output_size = 5

model = CustomLayer(input_size, hidden_size, output_size)
print(model)

## Second model
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
        
        # 第一层：输入层到隐藏层
        self.layer1 = SimpleLayer(input_dim, hidden_dim)
        
        # 第二层：隐藏层到输出层
        self.layer2 = SimpleLayer(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)  # 第一层前向传播
        x = self.layer2(x)  # 第二层前向传播
        return x

# 实例化一个MLP模型
input_dim = 784  # 假设输入维度为784（例如，MNIST数据集）
hidden_dim = 128  # 隐藏层维度
output_dim = 10  # 输出维度（例如，10类分类问题）
model = MLP(input_dim, hidden_dim, output_dim)

# 打印模型结构
print(model)

## Third Model
import torch
import torch.nn as nn

class ComplexModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ComplexModel, self).__init__()

        # 创建一个CustomLayer实例
        self.custom_layer = CustomLayer(input_size, hidden_size, output_size)

        # 创建一个MLP实例
        self.mlp = MLP(hidden_size, hidden_size, output_size)

    def forward(self, x):
        x = self.custom_layer(x)
        x = self.mlp(x)
        return x

# 实例化一个ComplexModel
input_size = 784  # 假设输入维度为784
hidden_size = 128  # 隐藏层维度
output_size = 10  # 输出维度
model = ComplexModel(input_size, hidden_size, output_size)

# 打印模型结构
print(model)


'''
In PyTorch, the nn.Module class provides two methods to access the parameters of the model: parameters() and named_parameters(). Both methods can be used to iterate over all parameters of a model, but they differ in what is returned.
parameters() method: Returns an iterable generator where each element is a tensor representing a parameter of the model.
'''
model = ComplexModel()
for param in model.parameters():
    print(param)

'''
named_parameters() method: Returns an iterable generator where each element is a tuple containing the name of the parameter and the corresponding tensor.
In practical applications, the named_parameters() method is usually used because it can obtain the names of parameters at the same time, which is useful for debugging and visualizing model parameters.
'''
model = ComplexModel()
for name, param in model.named_parameters():
    print(f"Name: {name}, Parameter: {param}")


'''
The model and all its parameters can be moved to the GPU or CPU using the to() method. It attempts to move the model to the GPU if the device is available, otherwise it remains on the CPU. Move the model to GPU (if available):
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


## Customize layers
class CustomNormalization(nn.Module):
    def __init__(self, dim):
        super(CustomNormalization, self).__init__()
        self.dim = dim

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        std = x.std(dim=self.dim, keepdim=True)
        return (x - mean) / (std + 1e-8)

model.add_module('custom_normalization', CustomNormalization(1))


## Optimizer
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)

# Lost Functions
import torch.nn as nn

# MSE
outputs = model(inputs)  # 模型的预测输出
targets = labels  # 真实标签

mse_loss = nn.MSELoss()
loss = mse_loss(outputs, targets)

# Cross Entropy
outputs = model(inputs)
targets = torch.LongTensor(labels)  # 确保标签是整数
cross_entropy_loss = nn.CrossEntropyLoss()
loss = cross_entropy_loss(outputs, targets)

# Binary Cross Entropy
outputs = model(inputs)  # 输出已经是概率
targets = labels.float()  # 标签转换为float，二分类通常为0或1
bce_loss = nn.BCELoss()
loss = bce_loss(outputs, targets)

# Weighted Binary Cross Entropy
num_samples = [100, 1000]  # 类别0有100个样本，类别1有1000个样本
weights = [1 / num_samples[0], 1 / num_samples[1]]  # 计算类别权重
weighted_bce_loss = nn.BCEWithLogitsLoss(weight=torch.tensor(weights))
outputs = model(inputs)
labels = labels.float()  # 将标签转换为浮点数，因为BCEWithLogitsLoss期望的是概率
loss = weighted_bce_loss(outputs, labels)

# Kullback-Leibler Divergence Loss - 假设有两个概率分布，preds是模型预测的概率分布，targets是实际的概率分布
preds = torch.randn(3, 5).softmax(dim=1)  # 预测概率分布，使用softmax转换
targets = torch.randn(3, 5).softmax(dim=1)  # 真实概率分布
# 初始化KLDivLoss实例，reduction参数定义了损失的聚合方式，可以是'mean'、'sum'或'none'
criterion = nn.KLDivLoss(reduction='batchmean')  # 'batchmean'表示对批量数据求平均
# 计算KLDivLoss
loss = criterion(preds.log(), targets)  # 注意：preds应该取对数，因为KLDivLoss默认期望log_softmax的输出
print('KLDivLoss:', loss.item())


# TripletMarginLoss
# 设置随机种子以获得可复现的结果
torch.manual_seed(42)
# 假设特征维度
feature_dim = 128
num_triplets = 10
anchors = torch.randn(num_triplets, feature_dim)
positives = torch.randn(num_triplets, feature_dim)
negatives = torch.randn(num_triplets, feature_dim)
# 将它们组合成形状为 (num_triplets, 3, feature_dim) 的张量
triplets = torch.stack((anchors, positives, negatives), dim=1)
# 初始化三元组损失函数，设置边际值m
triplet_loss = nn.TripletMarginLoss(margin=1.0)
# 计算损失
loss = triplet_loss(triplets[:, 0], triplets[:, 1], triplets[:, 2])
print('Triplet Margin Loss:', loss.item())

# Training models
output = model(inputs)
loss = criterion(output, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()


## Learning Rate
'''
StepLR is a basic and commonly used learning rate adjustment strategy, which adjusts the learning rate according to a predetermined period (usually calculated by epoch). Among them, the step_size parameter specifies the time interval at which decay occurs. For example, if step_size=5, the learning rate will be adjusted every 5 epochs. The gamma parameter controls the decay ratio of the learning rate with each adjustment. If gamma=0.1, this means that every time a step_size is reached, the learning rate will be multiplied by 0.1, which means it will decay to 10% of the original value.
'''
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 初始化模型和优化器
model = YourModel()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# 创建 StepLR 调度器
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# 训练循环
for epoch in range(num_epochs):
    # 训练过程...

    # 在每个epoch结束时调用scheduler的step()方法来更新学习率
    scheduler.step()

'''
MultiStepLR is similar to StepLR in that it adjusts the learning rate based on the period (epoch), but provides more flexible decay time point control. Through the milestones parameter, you can specify exactly at which specific epoch numbers the learning rate should decrease. The gamma parameter determines the proportion by which the learning rate decreases in each of these specified epochs.
For example, if you set milestones=[10, 20, 30] and gamma=0.1, then:
After the 10th epoch of training, the learning rate will be multiplied by 0.1 for the first time, that is, reduced to 10% of the original value.
Then, after the 20th epoch, the learning rate is multiplied by 0.1 again and decays to 1% of the original value relative to the initial value.
Finally, after the 30th epoch, the learning rate is multiplied by 0.1 again, and finally decays to 0.1% of the original value relative to the initial value.
This approach allows for more fine-grained control over the timing of the learning rate decrease based on performance changes during training or expected learning curves, helping the model converge better or avoid overfitting. Here is simple example code using MultiStepLR:
'''
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

# 初始化模型和优化器
model = YourModel()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# 创建 MultiStepLR 调度器
scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

# 训练循环
for epoch in range(num_epochs):
    # 训练过程...

    # 每个epoch结束时调用scheduler的step()方法检查是否需要调整学习率
    scheduler.step()


'''
ExponentialLR is a learning rate adjustment strategy that gradually decays the learning rate in an exponential function. Unlike StepLR and MultiStepLR, which suddenly change the learning rate at specific epochs, ExponentialLR gradually decreases the learning rate at a fixed rate after each training step (or each epoch, depending on the update frequency of the scheduler). This is useful for scenarios where the learning rate needs to be reduced smoothly to explore the solution space in more detail or to slowly approach the optimal solution later in training.
'''
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

# 初始化模型和优化器
model = YourModel()
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)

# 创建 ExponentialLR 调度器
# 其中 gamma 参数指定了学习率衰减的速率，例如 gamma=0.9 表示每经过一个调整周期，学习率变为原来的 90%
scheduler = ExponentialLR(optimizer, gamma=0.9)

# 训练循环
for epoch in range(num_epochs):
    # 训练过程...

    # 每个epoch结束时调用scheduler的step()方法更新学习率
    scheduler.step()


'''
Learning Rate Scheduler
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 假设的模型定义
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# 实例化模型和优化器
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 创建学习率调整器
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # 每5个epoch学习率减半

# 训练循环
num_epochs = 30
for epoch in range(num_epochs):
    # 假设的训练步骤，这里简化处理
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    # 每个epoch结束时更新学习率
    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], LR: {scheduler.get_last_lr()[0]}")

# Model Evaluation
model.eval()

with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        # 进行后续处理..

# accuracy
import torch

# 假设 outputs 和 labels 都是形状为 (batch_size,) 的张量
# 对于多分类问题，outputs 通常包含每个类别的概率，需要获取预测类别
_, predicted = torch.max(outputs.data, 1)

# 将预测和真实标签转换为相同的数据类型（例如，都转为整数）
if isinstance(predicted, torch.Tensor):
    predicted = predicted.cpu().numpy()
if isinstance(labels, torch.Tensor):
    labels = labels.cpu().numpy()

# 计算并输出准确率
accuracy = (predicted == labels).sum() / len(labels)
print(f'Accuracy: {accuracy}')

# precision
from sklearn.metrics import precision_score

# 假设 predictions 和 true_labels 是形状为 (n_samples,) 的数组
# predictions 是模型的预测结果，true_labels 是对应的真值标签
# 如果是多分类问题，labels 参数是所有类别的列表

# 二分类问题
precision_binary = precision_score(true_labels, predictions)

# 多分类问题，宏平均
precision_macro = precision_score(true_labels, predictions, average='macro')

# 多分类问题，微平均
precision_micro = precision_score(true_labels, predictions, average='micro')

print(f'Binary Precision: {precision_binary}')
print(f'Macro Average Precision: {precision_macro}')
print(f'Micro Average Precision: {precision_micro}')

# Recall
from sklearn.metrics import recall_score

# 假设 predictions 和 true_labels 是形状为 (n_samples,) 的数组
# predictions 是模型的预测结果，true_labels 是对应的真值标签
# 对于多分类问题，labels 参数是所有类别的列表

# 二分类问题
recall_binary = recall_score(true_labels, predictions)

# 多分类问题，宏平均
recall_macro = recall_score(true_labels, predictions, average='macro')

# 多分类问题，微平均
recall_micro = recall_score(true_labels, predictions, average='micro')

print(f'Binary Recall: {recall_binary}')
print(f'Macro Average Recall: {recall_macro}')
print(f'Micro Average Recall: {recall_micro}')

# F1 Score
from sklearn.metrics import f1_score

# 假设 predictions 和 true_labels 是形状为 (n_samples,) 的数组
# predictions 是模型的预测结果，true_labels 是对应的真值标签
# 对于多分类问题，labels 参数是所有类别的列表

# 二分类问题
f1_binary = f1_score(true_labels, predictions)

# 多分类问题，宏平均
f1_macro = f1_score(true_labels, predictions, average='macro')

# 多分类问题，微平均
f1_micro = f1_score(true_labels, predictions, average='micro')

print(f'Binary F1 Score: {f1_binary}')
print(f'Macro Average F1 Score: {f1_macro}')
print(f'Micro Average F1 Score: {f1_micro}')

# AUC
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 假设 y_true 是二分类的真值标签，y_scores 是模型的得分或概率输出
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]  # 真实标签
y_scores = [0.1, 0.4, 0.35, 0.8, 0.6, 0.1, 0.9, 0.7, 0.2, 0.5]  # 模型得分

# 计算ROC曲线的点
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# 计算ROC曲线下面积（AUC）
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # 绘制随机猜测的ROC曲线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")


# save model
# 保存状态字典
model = SimpleModel()
torch.save(model.state_dict(), 'model.pth')

# load model
model = ComplexModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('model.pth'))

# 加载状态字典
loaded_state_dict = torch.load('model_state.pth')

# 创建相同结构的新模型
new_model = SimpleModel()
# 将加载的状态字典加载到新模型
new_model.load_state_dict(loaded_state_dict)

# 加载整个模型
loaded_model = torch.load('model.pth')

# 在CPU上加载GPU上保存的模型
loaded_model = torch.load('model.pth', map_location=torch.device('cpu'))


# 保存整个模型（包括结构和参数）
torch.save(model, 'model.pth')

# 加载整个模型
loaded_model = torch.load('model.pth')

# 保存模型结构
torch.save(model, 'model_structure.pth', save_model_obj=False)

# 加载模型结构
loaded_model_structure = torch.load('model_structure.pth')

# 保存模型参数
torch.save(model.state_dict(), 'model_parameters.pth')

# 加载模型参数
loaded_parameters = torch.load('model_parameters.pth')
model.load_state_dict(loaded_parameters)
