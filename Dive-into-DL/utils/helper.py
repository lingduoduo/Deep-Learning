# utils/helper.py

import inspect
from typing import Iterable, Optional, Any, Tuple
import collections

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from IPython import display

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

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Default L2 loss assuming matching shapes."""
        return (y_hat - y) ** 2 / 2

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("forward() not implemented")

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return super().__call__(X)

    def training_step(self, batch) -> torch.Tensor:
        """Default training step: compute loss on a batch (X, y)."""
        X, y = batch
        l = self.loss(self(X), y)
        return l.mean()

    def validation_step(self, batch) -> torch.Tensor:
        """Optional validation step; by default, same as training loss."""
        X, y = batch
        l = self.loss(self(X), y)
        return l.mean()


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


# from utils.helper import (
#     HyperParameters, Module, add_to_class,
#     SyntheticRegressionData, Trainer,
#     LinearRegressionScratch, LinearRegression
# )

# # Scratch version
# model_scratch = LinearRegressionScratch(num_inputs=2, lr=0.03)
# data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
# trainer = Trainer(max_epochs=3)
# trainer.fit(model_scratch, data)

# # High-level nn.Module version
# model_nn = LinearRegression(num_inputs=2, lr=0.03)
# trainer = Trainer(max_epochs=3)
# trainer.fit(model_nn, data)
# w, b = model_nn.get_w_b()
# print("learned w:", w, "learned b:", b)


# @add_to_class(Trainer)
# def prepare_batch(self, batch):
#     # for future more complex data
#     return batch

