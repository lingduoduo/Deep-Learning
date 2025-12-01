from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import glob

# Class names for CIFAR-10
label_name = ["airplane", "automobile", "bird",
              "cat", "deer", "dog",
              "frog", "horse", "ship", "truck"]

# Map class names to numerical labels
label_dict = {name: idx for idx, name in enumerate(label_name)}

# Default image loader
def default_loader(path):
    return Image.open(path).convert("RGB")

# Data augmentation and preprocessing for training
train_transform = transforms.Compose([
    transforms.RandomCrop(28),   # Randomly crop the image to 28x28 pixels
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),  # Normalize with mean and std for RGB channels
])

# Preprocessing for testing (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# Alternative (commented out) data augmentation example
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop((28, 28)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     # transforms.RandomRotation(90),
#     transforms.RandomGrayscale(0.2),
#     transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
#     transforms.ToTensor()
# ])
#
# test_transform = transforms.Compose([
#     transforms.Resize((28, 28)),
#     transforms.ToTensor()
# ])

class MyDataset(Dataset):
    """
    Custom dataset for loading CIFAR-10 images from folders.

    Expected directory structure:
        /path/to/CIFAR10/TRAIN/<class_name>/*.png
        /path/to/CIFAR10/TEST/<class_name>/*.png
    """
    def __init__(self, im_list, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = []
        for im_item in im_list:
            # Example path:
            # "/home/ling/dataset/CIFAR10/TRAIN/airplane/aeroplane_s_000021.png"
            im_label_name = im_item.split("/")[-2]
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)
        if self.transform is not None:
            im_data = self.transform(im_data)
        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


# Load image paths
im_train_list = glob.glob("/home/ling/dataset/CIFAR10/TRAIN/*/*.png")
im_test_list = glob.glob("/home/ling/dataset/CIFAR10/TEST/*/*.png")

# Create datasets
train_dataset = MyDataset(im_train_list, transform=train_transform)
test_dataset = MyDataset(im_test_list, transform=test_transform)

# Create dataloaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=128,
                         shuffle=False,
                         num_workers=4)

print("Number of training images:", len(train_dataset))
print("Number of testing images:", len(test_dataset))
