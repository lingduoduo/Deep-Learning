import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
import os

# Define whether to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parser for command-line options (Linux-style)
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf',
                    default='./model/',
                    help='folder to output images and model checkpoints')  # Path to save outputs
args = parser.parse_args()

# Hyperparameters
EPOCH = 135       # Number of epochs
pre_epoch = 0     # Number of epochs already completed (for resuming training)
BATCH_SIZE = 128  # Batch size
LR = 0.01         # Learning rate

# Prepare dataset and preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # Pad with 0s and then randomly crop to 32x32
    transforms.RandomHorizontalFlip(),      # Randomly flip images horizontally
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),  # Normalize with mean and std for R, G, B
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/home/kuan/dataset/cifar-11-batches-py',
    train=True,
    download=False,
    transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='/home/kuan/dataset/cifar-11-batches-py',
    train=False,
    download=True,
    transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

# CIFAR-10 class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model definition - ResNet18
net = ResNet18().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross entropy loss (common for multi-class problems)
optimizer = optim.SGD(net.parameters(), lr=LR,
                      momentum=0.9, weight_decay=5e-4)  # Momentum SGD + L2 regularization

# Training
if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85  # Initialize best test accuracy
    print("Start Training, ResNet-18!")

    with open("acc.txt", "w") as f:
        with open("log.txt", "w") as f2:
            for epoch in range(pre_epoch, EPOCH):
                print(f'\nEpoch: {epoch + 1}')
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0

                for i, data in enumerate(trainloader, 0):
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # Forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # Print loss and accuracy for each batch
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length),
                             sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length),
                                sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # Test after each epoch
                print("Testing...")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    acc = 100. * correct / total
                    print('Test Accuracy: %.3f%%' % acc)

                    # Save model
                    print('Saving model...')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d, Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()

                    # Save best accuracy
                    if acc > best_acc:
                        with open("best_acc.txt", "w") as f3:
                            f3.write("EPOCH=%d, best_acc= %.3f%%" % (epoch + 1, acc))
                        best_acc = acc
            print("Training Finished, Total EPOCH=%d" % EPOCH)
