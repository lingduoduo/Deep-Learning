# utils/helper.py

import os
import re
import hashlib
import urllib.request

import inspect
from typing import Iterable, Optional, Any, Tuple
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
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
        self.save_hyperparameters()
        self.lr = lr
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

class Vocab:  # @save
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        tokens = [] if tokens is None else tokens
        reserved_tokens = [] if reserved_tokens is None else reserved_tokens

        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]

        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # Keep a stable order: <unk>, reserved tokens, then by frequency
        self.idx_to_token = ['<unk>'] + list(reserved_tokens)
        for token, freq in self.token_freqs:
            if freq < min_freq:
                break
            if token not in self.idx_to_token:
                self.idx_to_token.append(token)

        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    @property
    def unk(self):
        return self.token_to_idx['<unk>']

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[int(indices)]


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

