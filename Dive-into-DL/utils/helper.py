# utils/helper.py

import os
import re
import hashlib
import urllib.request
import zipfile
import requests
import inspect

from typing import Iterable, Optional, Any, Tuple, List
import collections
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from IPython import display

plt.figurefigsize=(6, 3)

import torchvision
from torchvision import transforms


# Chapter 2

def add_to_class(Class):
    """Decorator: add a function as a method to an existing class.

    Example:
        @add_to_class(MyClass)
        def new_method(self, x):
            ...
    """
    def wrapper(func):
        setattr(Class, func.__name__, func)
        return func
    return wrapper



class HyperParameters:
    """Base class that auto-saves __init__ arguments into `self.hparams`."""

    def save_hyperparameters(self, ignore=None):
        if ignore is None:
            ignore = []
        frame = inspect.currentframe().f_back
        args = frame.f_locals
        self.hparams = {}
        for k, v in args.items():
            if k != "self" and k not in ignore and not k.startswith("_"):
                setattr(self, k, v)
                self.hparams[k] = v


class Module(torch.nn.Module, HyperParameters):
    """Minimal D2L-like Module with a default loss & training_step."""

    def __init__(self):
        super().__init__()

    def loss(self, y_hat, y):
        return (y_hat - y) ** 2 / 2

    def forward(self, X):
        raise NotImplementedError

    def training_step(self, batch):
        X, y = batch
        return self.loss(self(X), y).mean()

    def validation_step(self, batch):
        X, y = batch
        return self.loss(self(X), y).mean()

    def plot(self, name, value, train=True):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if not hasattr(self, "history"):
            self.history = {"train": {}, "val": {}}
        key = "train" if train else "val"
        self.history[key].setdefault(name, []).append(value)
    
    def layer_summary(self, input_shape):
        """
        Print each layer's output shape.
        Works with Lazy layers (runs a dummy forward).
        """
        if not hasattr(self, "net"):
            raise AttributeError("layer_summary requires self.net")

        self.eval()
        X = torch.zeros(*input_shape)

        with torch.no_grad():
            for layer in self.net:
                X = layer(X)
                print(f"{layer.__class__.__name__:25s} -> {tuple(X.shape)}")

class SGD(HyperParameters):
    """Minibatch stochastic gradient descent for a list of tensors."""

    def __init__(self, params: Iterable[torch.Tensor], lr: float):
        self.lr = lr
        self.save_hyperparameters()
        # `params` is stored by save_hyperparameters as self.params

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class DataModule(HyperParameters):  # @save
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train: bool):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train: bool, indices=None):
        """Utility: create a DataLoader from a list of tensors.

        tensors: list/tuple of tensors with same first dimension
        train:   whether to shuffle
        indices: slice or index array to select subset
        """
        if indices is None:
            indices = slice(None)

        sliced = [t[indices] for t in tensors]
        dataset = TensorDataset(*sliced)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=train)


class SyntheticRegressionData(DataModule):  # @save
    """Generate y = X w + b + small noise for linear regression."""

    def __init__(
        self,
        w: torch.Tensor,
        b: float,
        num_train: int = 1000,
        num_val: int = 100,
        batch_size: int = 32,
    ):
        # Make sure w is column vector
        w = w.reshape(-1, 1)
        self.save_hyperparameters()

        num_inputs = w.shape[0]

        # Train data
        X_train = torch.randn(num_train, num_inputs)
        y_train = X_train @ w + b
        y_train += 0.01 * torch.randn_like(y_train)

        # Val data
        X_val = torch.randn(num_val, num_inputs)
        y_val = X_val @ w + b
        y_val += 0.01 * torch.randn_like(y_val)

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

    def get_dataloader(self, train: bool):
        if train:
            ds = TensorDataset(self.X_train, self.y_train)
        else:
            ds = TensorDataset(self.X_val, self.y_val)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=train)


class Trainer(HyperParameters):  # @save
    """Minimal trainer that relies on model.configure_optimizers().

    Expected model interface:
        - model(X): forward pass
        - model.training_step(batch): returns a scalar loss tensor
        - model.configure_optimizers(): returns an optimizer-like object
          with .zero_grad() and .step()
    """

    def __init__(self, max_epochs: int = 3, gradient_clip_val: float = 0.0):
        self.save_hyperparameters()

    def fit(self, model: Module, data: DataModule):
        self.model = model
        self.data = data
        self.model.train()

        # Ask the model for its optimizer (works for scratch + nn versions)
        self.optim = self.model.configure_optimizers()

        train_iter = data.train_dataloader()
        val_iter = data.val_dataloader() if hasattr(data, "val_dataloader") else None

        for epoch in range(self.max_epochs):
            # Training loop
            train_losses = []
            for batch in train_iter:
                loss = self.model.training_step(batch)
                self.optim.zero_grad()
                loss.backward()

                if self.gradient_clip_val and self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)

                self.optim.step()
                train_losses.append(loss.item())

            # Validation (optional)
            val_loss = None
            if val_iter is not None:
                self.model.eval()
                with torch.no_grad():
                    val_losses = []
                    for batch in val_iter:
                        l = self.model.validation_step(batch)
                        val_losses.append(l.item())
                val_loss = sum(val_losses) / len(val_losses)
                self.model.train()

            # Logging
            train_loss = sum(train_losses) / len(train_losses)
            if val_loss is not None:
                print(
                    f"epoch {epoch + 1}, train loss {train_loss:.6f}, "
                    f"val loss {val_loss:.6f}"
                )
            else:
                print(f"epoch {epoch + 1}, train loss {train_loss:.6f}")

    @staticmethod
    def clip_gradients(max_norm: float, model: Module):
        """Gradient clipping by global norm."""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# Chapter 3

class LinearRegressionScratch(Module):  # @save
    """The linear regression model implemented from scratch."""

    def __init__(self, num_inputs: int, lr: float, sigma: float = 0.01):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l = (y_hat - y) ** 2 / 2
        return l.mean()

    def configure_optimizers(self):
        # Use custom SGD on [w, b]
        return SGD([self.w, self.b], self.lr)


class LinearRegression(Module):  # @save
    """Linear regression model implemented with high-level PyTorch APIs."""

    def __init__(self, num_inputs: int, lr: float):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        # We can use nn.Linear since num_inputs is known
        self.net = nn.Linear(num_inputs, 1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def get_w_b(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.net.weight.data, self.net.bias.data


class ProgressBoard(HyperParameters):
    """Simple D2L-style progress board for live plotting."""

    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear',
                 fig=None, axes=None, figsize=(3.5, 2.5), display_=True):
        self.save_hyperparameters()
        if fig is None or axes is None:
            self.fig, self.axes = plt.subplots(figsize=self.figsize)
        else:
            self.fig, self.axes = fig, axes

        self.raw_points = collections.defaultdict(list)   # label -> [(x, y), ...]
        self.data = collections.defaultdict(list)         # label -> [(x, y), ...]


    def draw(self, x, y, label, every_n=1):
        """Add a point and update the plot every `every_n` samples."""
        self.raw_points[label].append((x, y))
        if len(self.raw_points[label]) < every_n:
            return

        # Average the buffered points
        xs, ys = zip(*self.raw_points[label])
        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)
        self.data[label].append((avg_x, avg_y))
        self.raw_points[label].clear()

        if not self.display_:
            return

        ax = self.axes
        ax.cla()

        # Plot all series
        for i, (lbl, pts) in enumerate(self.data.items()):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, label=lbl)

        # Axes settings
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)

        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)

        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)
        ax.legend()
        self.fig.tight_layout()

        display.display(self.fig)
        display.clear_output(wait=True)


def plot(X, Y=None, xlabel=None, ylabel=None,
         legend=None, xscale=None, yscale=None,
         figsize=(6, 3), xlim=None):
    # If only X is given, plot X against its index (Zipf plot case)
    if Y is None:
        X = np.asarray(X)
        plt.plot(np.arange(len(X)), X)
    else:
        if isinstance(Y, (list, tuple)):
            for y in Y:
                plt.plot(X, y)
        else:
            plt.plot(X, Y)

    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)

    if legend:
        plt.legend(legend)

    if xlim is not None:
        plt.xlim(xlim)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Chapter 4

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            img = img.numpy()
        # For grayscale images
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if titles and i < len(titles):
            ax.set_title(titles[i])
    plt.tight_layout()
    return axes

class FashionMNIST(DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
    
    def get_dataloader(self, train: bool):
        """Return the train or validation dataloader."""
        dataset = self.train if train else self.val
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,          # shuffle only for training
            num_workers=self.num_workers,
        )
    
    def text_labels(self, indices):
        """Return text labels."""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]
    
    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1), nrows, ncols, titles=labels)


class Classifier(Module):  # @save
    """The base class of classification models."""

    # ---- core classifier behavior ----
    def loss(self, y_hat, y):
        # y_hat: (B, C), y: (B,)
        return F.cross_entropy(y_hat, y)

    def accuracy(self, y_hat, y):
        preds = y_hat.argmax(dim=1)
        return (preds == y).float().mean()

    def training_step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.plot("loss", loss, train=True)
        self.plot("acc", acc, train=True)
        return loss  # scalar tensor

    def validation_step(self, batch):
        """Validation step used by all classifiers."""
        X, y = batch
        y_hat = self(X)

        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.plot("loss", loss, train=False)
        self.plot("acc", acc, train=False)
        return loss

    # ---- utilities you already had ----
    def apply_init(self, inputs, init_fn=None):
        self.eval()
        with torch.no_grad():
            if isinstance(inputs, (list, tuple)):
                _ = self(*inputs)
            else:
                _ = self(inputs)

        if init_fn is not None:
            if hasattr(self, "net"):
                self.net.apply(init_fn)
            else:
                self.apply(init_fn)
        return self

    def save_hyperparameters(self, ignore=()):
        frame = inspect.currentframe().f_back
        args, _, _, local_vars = inspect.getargvalues(frame)
        ignore = set(ignore)
        self.hparams = {
            k: local_vars[k]
            for k in args
            if k != "self" and k not in ignore
        }
        return self


class SoftmaxRegressionScratch(Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)
    
    def forward(self, X):
        X = X.reshape(X.shape[0], -1)
        return X @ self.W + self.b  # logits

    def parameters(self):
        return [self.W, self.b]

    def parameters(self):
        return [self.W, self.b]
    
    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)
    
    def configure_optimizers(self):
        return SGD(self.parameters(), self.lr)
    
    def validation_step(self, batch):
        X, y = batch
        y_hat = self(X)
        return self.loss(y_hat, y)

class SoftmaxRegression(Classifier):  # @save
    def __init__(self, num_outputs: int, lr: float):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        # High-level network: flatten then linear layer
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_outputs)
        )

    def forward(self, X):
        # X: (batch, 1, 28, 28) -> net -> logits (batch, num_outputs)
        return self.net(X)

    def loss(self, y_hat, y):
        # Cross-entropy on logits
        return F.cross_entropy(y_hat, y)

    def configure_optimizers(self):
        # Use standard PyTorch SGD on all trainable parameters
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def validation_step(self, batch):
        """
        Validation step that returns loss (for Trainer),
        and can also log metrics if plot/accuracy are available.
        """
        X, y = batch
        y_hat = self(X)
        loss = self.loss(y_hat, y)

        return loss

# Chapter 5

class MLPScratch(Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        # Parameters of a 1-hidden-layer MLP
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, X):
        # X: (batch, 1, 28, 28) -> flatten -> hidden -> logits
        X = X.reshape(X.shape[0], -1)        # (batch, num_inputs)
        H = torch.relu(X @ self.W1 + self.b1)  # (batch, num_hiddens)
        out = H @ self.W2 + self.b2            # (batch, num_outputs)
        return out

    def loss(self, y_hat, y):
        # Cross-entropy on logits
        return F.cross_entropy(y_hat, y)

    def configure_optimizers(self):
        # Use standard PyTorch SGD on all trainable parameters
        return SGD(self.parameters(), lr=self.lr)

    def validation_step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = self.loss(y_hat, y)
        return loss


class MLP(Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_hiddens),
            nn.ReLU(),
            nn.LazyLinear(num_outputs)
        )

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def validation_step(self, batch):
        X, y = batch
        return self.loss(self(X), y)

class DropoutMLPScratch(Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()  # saves lr, dropout_1, dropout_2, etc.

        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        # Flatten: (batch, 1, 28, 28) -> (batch, 784)
        X = X.reshape(X.shape[0], -1)

        H1 = self.relu(self.lin1(X))
        H1 = F.dropout(H1, p=self.dropout_1, training=self.training)

        H2 = self.relu(self.lin2(H1))
        H2 = F.dropout(H2, p=self.dropout_2, training=self.training)

        return self.lin3(H2)

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)

    def validation_step(self, batch):
        X, y = batch
        y_hat = self(X)
        return self.loss(y_hat, y)
    
class DropoutMLP(Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_hiddens_1),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.LazyLinear(num_hiddens_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.LazyLinear(num_outputs),
        )

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)

    def validation_step(self, batch):
        X, y = batch
        return self.loss(self(X), y)
   
# Chapter 7

def corr2d(X, K):  
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

class LeNet(Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes),
        )

    def forward(self, X):
        return self.net(X)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams["lr"])

def init_cnn(m):
    # Conv: normal + lazy
    if isinstance(m, (nn.Conv2d, nn.LazyConv2d)):
        # after a dummy forward, m.weight is a real Parameter
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # Linear: normal + lazy
    elif isinstance(m, (nn.Linear, nn.LazyLinear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResNeXtBlock(nn.Module):  
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)

# Chapter 9

class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']


class TimeMachine(DataModule):
    """The Time Machine dataset."""

    def __init__(self, batch_size=32, num_steps=35, num_train=10000, num_val=5000,
                 root="./data", token="char", min_freq=0, reserved_tokens=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val
        self.root = root

        # IMPORTANT: set token BEFORE build() is called
        self.token = token  # "char" or "word"
        self.min_freq = min_freq
        self.reserved_tokens = [] if reserved_tokens is None else reserved_tokens

        corpus, self.vocab = self.build(self._download())

        # Create training samples: X is length num_steps, Y is X shifted by 1
        array = torch.tensor(
            [corpus[i:i + num_steps + 1] for i in range(len(corpus) - num_steps)],
            dtype=torch.long
        )
        self.X, self.Y = array[:, :-1], array[:, 1:]

    def _download(self):
        DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
        fname = "timemachine.txt"
        sha1 = "090b5e7e70c295757f55df93cb0a180b9691891a"

        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, fname)
        url = DATA_URL + fname

        # Download if needed
        if not os.path.exists(path):
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, path)

        # Verify SHA-1 checksum
        with open(path, "rb") as f:
            data = f.read()
            hash_val = hashlib.sha1(data).hexdigest()
            if hash_val != sha1:
                raise RuntimeError(f"SHA-1 mismatch: expected {sha1}, got {hash_val}")

        # Return raw text
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+', ' ', text).lower()

    def _tokenize(self, text: str):
        token = getattr(self, "token", "char")
        if token == "word":
            return text.split()
        return list(text)

    def build(self, raw_text: str, vocab=None):
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None:
            vocab = Vocab(tokens, min_freq=self.min_freq, reserved_tokens=self.reserved_tokens)
        corpus = vocab[tokens]  # list[int]
        return corpus, vocab

    def train_dataloader(self):
        i = slice(0, min(self.num_train, len(self.X)))
        ds = TensorDataset(self.X[i], self.Y[i])
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        start = min(self.num_train, len(self.X))
        end = min(self.num_train + self.num_val, len(self.X))
        ds = TensorDataset(self.X[start:end], self.Y[start:end])
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

class RNN(Module):  
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)

class RNNScratch(Module): 
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = nn.Parameter(
            torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))
        

class RNNLMScratch(Classifier):
    """The RNN-based language model implemented from scratch (no @add_to_class)."""

    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.rnn = rnn
        self.vocab_size = vocab_size
        self.lr = lr
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        self.W_hq = nn.Parameter(
            torch.randn(self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
        )
        self.b_q = nn.Parameter(torch.zeros(self.vocab_size))

    def one_hot(self, X):
        # X: (batch_size, num_steps)
        # Return: (num_steps, batch_size, vocab_size)
        return F.one_hot(X.T, self.vocab_size).to(torch.float32)

    def output_layer(self, rnn_outputs):
        # rnn_outputs: (num_steps, batch_size, num_hiddens) OR iterable of (batch, hidden)
        if isinstance(rnn_outputs, torch.Tensor):
            # (T, B, H) @ (H, V) -> (T, B, V) -> transpose -> (B, T, V)
            out = torch.matmul(rnn_outputs, self.W_hq) + self.b_q
            return out.transpose(0, 1)
        else:
            # list/tuple of (B, H) -> stack -> (B, T, V)
            outputs = [torch.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
            return torch.stack(outputs, dim=1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)

    def loss(self, y_hat, y):
        # y_hat: (B, T, V), y: (B, T)
        return F.cross_entropy(y_hat.reshape(-1, self.vocab_size), y.reshape(-1))

    def training_step(self, batch):
        X, y = batch
        l = self.loss(self(X), y)
        self.plot('ppl', torch.exp(l), train=True)
        return l  # keep returning loss

    def validation_step(self, batch):
        X, y = batch
        l = self.loss(self(X), y)
        self.plot('ppl', torch.exp(l), train=False)
        return l  

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
    
    def predict(self, prefix, num_preds, vocab, device=None):
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        state = None
        outputs = [vocab[prefix[0]]]

        with torch.no_grad():
            for i in range(len(prefix) + num_preds - 1):
                # last generated token id -> shape (B=1, T=1)
                X = torch.tensor([[outputs[-1]]], device=device, dtype=torch.long)

                # one-hot -> (T=1, B=1, V)
                embs = self.one_hot(X)

                # rnn -> rnn_outputs: (T=1, B=1, H)
                rnn_outputs, state = self.rnn(embs, state)

                if i < len(prefix) - 1:
                    # warm-up: force next char from prefix
                    outputs.append(vocab[prefix[i + 1]])
                else:
                    # predict next token
                    Y = self.output_layer(rnn_outputs)   # (B=1, T=1, V)
                    next_id = int(Y.argmax(dim=2).reshape(-1)[0])
                    outputs.append(next_id)

        return ''.join([vocab.idx_to_token[i] for i in outputs])

class RNNLM(RNNLMScratch):  
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)

    def output_layer(self, hiddens):
        return self.linear(hiddens).swapaxes(0, 1)

## Chapter 10

class LSTMScratch(Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        # IMPORTANT: RNNLMScratch expects these attributes
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma

        def init_weight(*shape):
            return nn.Parameter(torch.randn(*shape) * sigma)

        def triple():
            return (
                init_weight(num_inputs, num_hiddens),
                init_weight(num_hiddens, num_hiddens),
                nn.Parameter(torch.zeros(num_hiddens)),
            )

        self.W_xi, self.W_hi, self.b_i = triple()  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple()  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple()  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple()  # Candidate cell

    def forward(self, inputs, H_C=None):
        # inputs: (T, B, V)
        T, B, _ = inputs.shape

        if H_C is None:
            H = torch.zeros((B, self.num_hiddens), device=inputs.device)
            C = torch.zeros((B, self.num_hiddens), device=inputs.device)
        else:
            H, C = H_C

        outputs = []
        for t in range(T):
            X = inputs[t]  # (B, V)

            I = torch.sigmoid(X @ self.W_xi + H @ self.W_hi + self.b_i)
            F = torch.sigmoid(X @ self.W_xf + H @ self.W_hf + self.b_f)
            O = torch.sigmoid(X @ self.W_xo + H @ self.W_ho + self.b_o)
            C_tilde = torch.tanh(X @ self.W_xc + H @ self.W_hc + self.b_c)

            C = F * C + I * C_tilde
            H = O * torch.tanh(C)

            outputs.append(H)

        # (T, B, H)
        return torch.stack(outputs, dim=0), (H, C)

class LSTM(Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma

        self.rnn = nn.LSTM(num_inputs, num_hiddens)

    def forward(self, inputs, H_C=None):
        # inputs: (T, B, V)
        return self.rnn(inputs, H_C)
    
class GRU(RNN):
    def __init__(self, num_inputs, num_hiddens):
        Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens)


class StackedRNNScratch(Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = nn.ModuleList([
            LSTMScratch(num_inputs if i == 0 else num_hiddens,
                        num_hiddens, sigma)
            for i in range(num_layers)
        ])

    def forward(self, inputs, Hs=None):
        if Hs is None:
            Hs = [None] * self.num_layers

        X = inputs
        new_Hs = []
        for i, rnn in enumerate(self.rnns):
            X, Hi = rnn(X, Hs[i])   # now valid: LSTMScratch takes (inputs, state)
            new_Hs.append(Hi)

        return X, new_Hs


class Encoder(nn.Module):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(Classifier):  #@save
    """The base class for the encoder--decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]
    
## Chapter NLP

def _sha1sum(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def download_and_extract_ptb(root: str = "./data") -> str:
    """
    Download ptb.zip to `root`, verify SHA1, and extract.
    Returns the extracted PTB directory (e.g., ./data/ptb).
    """
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "ptb.zip")
    ptb_dir = os.path.join(root, "ptb")

    # Download if missing
    if not os.path.exists(zip_path):
        print(f"Downloading PTB to {zip_path} ...")
        r = requests.get(PTB_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)

    # Verify checksum
    if _sha1sum(zip_path) != PTB_SHA1:
        raise RuntimeError("PTB zip SHA1 mismatch. Please delete ptb.zip and re-download.")

    # Extract if missing
    if not os.path.exists(ptb_dir):
        print(f"Extracting PTB to {ptb_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root)

    return ptb_dir

def read_ptb(split: str = "train", root: str = "./data") -> List[List[str]]:
    """
    Load PTB split into a list of token lists (one list per line).
    split: 'train' | 'valid' | 'test'
    """
    ptb_dir = download_and_extract_ptb(root)
    fname = f"ptb.{split}.txt"
    path = os.path.join(ptb_dir, fname)
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return [line.split() for line in raw.split("\n") if line.strip() != ""]


class BiRNNScratch(Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.f_rnn = RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens = num_hiddens * 2  # output hidden size doubles

    def forward(self, inputs, Hs=None):
        f_outputs = self.f_rnn(inputs)  # list length T, each (B, H)
        b_inputs = torch.flip(inputs, dims=[0])
        b_outputs_rev = self.b_rnn(b_inputs)  # list in reversed-time order

        b_outputs = list(reversed(b_outputs_rev))
        outputs = [torch.cat((f, b), dim=-1) for f, b in zip(f_outputs, b_outputs)]
        return outputs, None

def download_and_extract_zip(url: str, root: str, zip_name: str, sha1: str) -> str:
    """
    Download a zip to `root/zip_name`, verify sha1, extract into `root`.
    Returns the extraction root directory (same as `root`).
    """
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, zip_name)

    # download if missing
    if not os.path.exists(zip_path):
        print(f"Downloading {zip_name} to {zip_path} ...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)

    # verify
    got = _sha1sum(zip_path)
    if got != sha1:
        raise RuntimeError(f"SHA1 mismatch for {zip_name}: expected {sha1}, got {got}")

    # extract (idempotent-ish; zipfile will overwrite files if they exist)
    print(f"Extracting {zip_name} into {root} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(root)

    return root

def init_seq2seq(module):  #@save
    """Initialize weights for sequence-to-sequence learning."""
    if type(module) == nn.Linear:
         nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])

class Seq2SeqEncoder(Encoder):  # @save
    """The RNN encoder for sequence-to-sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size,
            num_hiddens,
            num_layers,
            dropout=dropout if num_layers > 1 else 0
        )
        self.apply(init_seq2seq)

    def forward(self, X, *args):
        embs = self.embedding(X.t().type(torch.int64))  # (T, B, E)
        outputs, state = self.rnn(embs)
        return outputs, state

@add_to_class(EncoderDecoder)  #@save
def predict_step(self, batch, device, num_steps,
                 save_attention_weights=False):
    batch = [a.to(device) for a in batch]
    src, tgt, src_valid_len, _ = batch
    enc_all_outputs = self.encoder(src, src_valid_len)
    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    outputs, attention_weights = [tgt[:, (0)].unsqueeze(1), ], []
    for _ in range(num_steps):
        Y, dec_state = self.decoder(outputs[-1], dec_state)
        outputs.append(Y.argmax(2))
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weights.append(self.decoder.attention_weights)
    return torch.cat(outputs[1:], 1), attention_weights

def bleu(pred_seq, label_seq, k):  #@save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

class Seq2Seq(EncoderDecoder):  # @save
    """The RNN encoder--decoder for sequence-to-sequence learning."""
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_pad = tgt_pad
        self.lr = lr
        self.save_hyperparameters()

    def training_step(self, batch):
        # batch: (src, tgt_in, src_valid_len, tgt_out)
        Y_hat = self(*batch[:-1])          # (B, T, V)
        l = self.loss(Y_hat, batch[-1])    # masked CE
        self.plot('loss', l, train=True)
        return l


    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        l = self.loss(Y_hat, batch[-1])
        self.plot('loss', l, train=False)
        return l 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss(self, Y_hat, Y):
        """
        Masked cross-entropy loss.
        Y_hat: (B, T, V)
        Y:     (B, T)
        """
        import torch.nn.functional as F

        B, T, V = Y_hat.shape
        y_hat = Y_hat.reshape(-1, V)   # (B*T, V)
        y = Y.reshape(-1)              # (B*T,)

        # per-token CE
        l = F.cross_entropy(y_hat, y, reduction='none')  # (B*T,)

        # mask out <pad>
        mask = (y != self.tgt_pad).float()
        return (l * mask).sum() / mask.sum().clamp_min(1)



class MTFraEng(DataModule):  # @save
    """The English-French dataset."""
    def _download(self, DATA_URL="http://d2l-data.s3-accelerate.amazonaws.com/"):
        download_and_extract_zip(
            url=DATA_URL + "fra-eng.zip",
            root=self.root,
            zip_name="fra-eng.zip",
            sha1="94646ad1522d915e7b0f9296181140edcf86a4f5",
        )
        path = os.path.join(self.root, "fra-eng", "fra.txt")
        with open(path, encoding="utf-8") as f:
            return f.read()

@add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # Replace non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)

@add_to_class(MTFraEng)  #@save
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt

@add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())

@add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = Vocab(sentences, min_freq=2)
        array = torch.tensor([vocab[s] for s in sentences])
        valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
        return array, vocab, valid_len
    src, tgt = self._tokenize(self._preprocess(raw_text),
                              self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)

@add_to_class(MTFraEng) 
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays

@add_to_class(MTFraEng)  # @save
def get_dataloader(self, train: bool):
    src_array, tgt_in, src_valid_len, tgt_out = self.arrays

    n_train = self.num_train
    idx = slice(0, n_train) if train else slice(n_train, n_train + self.num_val)

    dataset = TensorDataset(
        src_array[idx],
        tgt_in[idx],
        src_valid_len[idx],
        tgt_out[idx],
    )
    return DataLoader(dataset, batch_size=self.batch_size, shuffle=train, drop_last=train)

def check_shape(x, shape):
    assert tuple(x.shape) == shape, \
        f"Expected shape {shape}, got {tuple(x.shape)}"

# Chapter 11

def show_heatmaps(matrices, xlabel, ylabel, titles=None,
                  figsize=(2.5, 2.5), cmap='Reds'):
    """Show heatmaps of matrices using matplotlib (no d2l)."""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Convert to numpy safely
    if isinstance(matrices, torch.Tensor):
        matrices = matrices.detach().cpu().numpy()
    else:
        matrices = np.array(matrices)

    num_rows, num_cols, _, _ = matrices.shape

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False
    )

    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i, j]
            pcm = ax.imshow(matrices[i, j], cmap=cmap, aspect='auto')

            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles is not None:
                ax.set_title(titles[j])

    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.tight_layout()
    plt.show()

def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    
class AdditiveAttention(nn.Module):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

class AttentionDecoder(Decoder): 
    """The base attention-based decoder interface."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


class MultiHeadAttention(Module):  #@save
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

@add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """Transposition for parallel computation of multiple attention heads."""
    # Shape of input X: (batch_size, no. of queries or key-value pairs,
    # num_hiddens). Shape of output X: (batch_size, no. of queries or
    # key-value pairs, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    # Shape of output X: (batch_size, num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3)
    # Shape of output: (batch_size * num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

@add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """Reverse the operation of transpose_qkv."""
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nn.Module):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(Encoder): 
    """The Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

class TransformerDecoderBlock(nn.Module):
    # The i-th block in the Transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
    
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))

class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))


class ViT(Classifier):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)