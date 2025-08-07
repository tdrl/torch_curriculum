"""Simplest possible torch demo: linear projection with integer values."""

from torch_playground.util import BaseArguments, App, save_tensor, get_default_working_dir
import torch
import torch.nn as nn
import torch.cuda
from torch.utils.data import TensorDataset
from torchinfo import summary
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
import json


class HRLinearTrainable(nn.Module):
    """A simple linear classifier, without bias; trainable.

    This model assumes that data points are rows in the input tensor.
    """

    def __init__(self, input_dim: int, W: Optional[torch.Tensor] = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert input_dim > 0, 'Input dimension must be positive.'
        self.input_dim = input_dim
        if W is None:
            w_tmp = torch.randn((input_dim,), dtype=dtype, requires_grad=True)
        else:
            assert W.shape == (input_dim,), 'Weight tensor shape mismatch.'
            w_tmp = W
        self.W = nn.Parameter(w_tmp, requires_grad=True)

    def __repr__(self):
        return f'HRLinearTrainable(input_dim={self.input_dim}, W.shape={self.W.shape})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layer."""
        return ((x @ self.W) > 0) * 2 - 1  # Returns -1 or 1 based on the sign of the linear combination.


@dataclass
class LinearTrainableArguments(BaseArguments):
    """Command line arguments for the linear trainable application."""
    dim: int = field(default=10, metadata=BaseArguments._meta(help='The dimension of the input features.'))
    num_train_samples: int = field(default=100, metadata=BaseArguments._meta(help='Number of training samples to generate.'))
    num_epochs: int = field(default=10, metadata=BaseArguments._meta(help='Number of epochs to train the model.'))
    learning_rate: float = field(default=0.01, metadata=BaseArguments._meta(help='Learning rate for the optimizer.'))
    output_dir: Path = field(default=get_default_working_dir(),
                             metadata=BaseArguments._meta(help='Directory to save the data, checkpoints, trained model, etc.'))

class LinearTrainableApp(App[LinearTrainableArguments]):
    """An application that trains a simple linear model."""

    def __init__(self, argv: Optional[list[str]] = None):
        super().__init__(LinearTrainableArguments(),
                         'Train a simple linear model to fit a separable problem in low-dimensional space.',
                         argv=argv)
        self.dtype = torch.float32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.debug('Assigned device', device=self.device)
        torch.manual_seed(self.args.randseed)
        self.logger.debug('Set random seed', randseed=self.args.randseed)

    def create_data(self) -> tuple[TensorDataset, torch.Tensor]:
        """Create a dataset of a d-dimensional discriminator, random inputs, and classified outputs.

        Data points are rows.

        Returns:
            TensorDataset: A dataset containing random integer inputs.
        """
        assert self.args.dim > 0, 'X dimension must be positive.'
        assert self.args.num_train_samples > 0, 'Number of samples must be positive.'
        self.logger.info('Creating data set', dim=self.args.dim, num_train_samples=self.args.num_train_samples)
        # A zero-mean, unit-variance, normally distributed data set.
        X = torch.randn(size=(self.args.num_train_samples, self.args.dim), dtype=self.dtype)
        self.logger.debug('X[0:3]', sample=X[:3])
        discriminator = torch.randn(size=(self.args.dim,), dtype=self.dtype)
        self.logger.debug('Discriminator', sample=discriminator)
        # Classify the data points based on the discriminator.
        y = ((X @ discriminator > 0) * 2 - 1).long()
        self.logger.debug('y[0:5]', sample=y[:5])
        self.logger.info('Data set created',
                         num_samples=X.shape[0],
                         dim=X.shape[1],
                         num_positive_samples=(y > 0).sum().item(),
                         num_negative_samples=(y < 0).sum().item())
        return TensorDataset(X, y), discriminator


    def run(self):
        # TODO(heather): This is common boilerplate, should be moved to App.
        try:
            self.logger.info('Starting LinearTrainableApp with arguments', **asdict(self.args))
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.args.output_dir / 'args.txt', 'w') as f:
                json.dump(asdict(self.args), f, indent=2, default=str)
            data, discriminator = self.create_data()
            save_tensor(discriminator, self.args.output_dir / 'discriminator')
            save_tensor(data.tensors[0], self.args.output_dir / 'X')
            save_tensor(data.tensors[1], self.args.output_dir / 'y')
            self.model = HRLinearTrainable(input_dim=self.args.dim, dtype=self.dtype).to(self.device)
            self.logger.info('Sample output', sample_output=self.model(data.tensors[0].to(self.device))[:5])
            self.logger.info('Model summary', model=summary(self.model, input_size=(self.args.num_train_samples, self.args.dim), verbose=0))
            for name, param in self.model.named_parameters():
                self.logger.debug('Parameter', name=name, shape=param.shape, requires_grad=param.requires_grad)
        except Exception as e:
            self.logger.exception('Uncaught error somewhere in the code (hopeless).', exc_info=e)
            raise


if __name__ == '__main__':
    LinearTrainableApp().run()
