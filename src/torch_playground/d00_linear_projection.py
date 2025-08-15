"""Simplest possible torch demo: linear projection with integer values."""

from torch_playground.util import setup_logging, save_tensor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchinfo import summary
import argparse
from typing import Optional
from pathlib import Path
from logging import Logger
from dataclasses import dataclass


class HRLinear(nn.Module):
    """A simple linear projection model.

    This model assumes that data points are rows in the input tensor.
    """

    def __init__(self, input_dim: int, output_dim: int, W: Optional[torch.Tensor] = None, b: Optional[torch.Tensor] = None):
        super(HRLinear, self).__init__()
        assert input_dim > 0, 'Input dimension must be positive.'
        assert output_dim > 0, 'Output dimension must be positive.'
        self.input_dim = input_dim
        self.output_dim = output_dim
        if W is None:
            self.W = torch.randint(-10, 10, (input_dim, output_dim), dtype=torch.int32)
        else:
            assert W.shape == (input_dim, output_dim), 'Weight tensor shape mismatch.'
            self.W = W
        if b is None:
            self.b = torch.randint(-10, 10, (output_dim,), dtype=torch.int32)
        else:
            assert b.shape == (output_dim,), 'Bias tensor shape mismatch.'
            self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layer."""
        return x @ self.W + self.b


def create_data(input_dim: int, n: int) -> TensorDataset:
    """Create a dataset of random integer inputs.

    Data points are rows.

    Args:
        input_dim (int): The dimension of the input features.
        n (int): The number of samples to generate.

    Returns:
        TensorDataset: A dataset containing random integer inputs.
    """
    assert input_dim > 0, 'Input dimension must be positive.'
    assert n > 0, 'Number of samples must be positive.'
    x = torch.randint(-10, 10, (n, input_dim), dtype=torch.int32)
    return TensorDataset(x)


@dataclass
class Arguments:
    """Command line arguments for the demo."""
    input_dim: int = 3
    output_dim: int = 3
    n_samples: int = 10
    output_dir: str = '/tmp/heather'  # TODO: better default


def parse_args() -> Arguments:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Linear projection demo with PyTorch.')
    parser.add_argument('--input_dim', type=int, default=3, help='Input dimension size.')
    parser.add_argument('--output_dim', type=int, default=3, help='Output dimension size.')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples to generate.')
    parser.add_argument('--output_dir', type=str, default='/tmp/heather', help='Directory to save output files.')  # TODO: better default
    return parser.parse_args(namespace=Arguments())


def main(logger: Logger, args: Arguments):
    output_dir = Path(args.output_dir) / 'd00_linear_projection'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Output directory: {output_dir}')
    model = HRLinear(input_dim=args.input_dim, output_dim=args.output_dim)
    logger.debug(f'Model = {model}')
    summary(model)
    torch.save(model.state_dict(), output_dir / 'model.pth')
    with open(output_dir / 'model.txt', 'w') as f:
        f.write(str(model))
    logger.debug(f'Model parameters: W={model.W}, b={model.b}')
    x = create_data(args.input_dim, args.n_samples)
    save_tensor(x.tensors[0], output_dir / 'x.txt')
    y = model(x.tensors[0])
    save_tensor(y, output_dir / 'y.txt')
    for i in range(args.n_samples):
        logger.debug(f'{i}: {x[i]} -> {i}: {y[i]}')


if __name__ == '__main__':
    logger = setup_logging()
    logger.info('Starting linear projection demo.')
    args = parse_args()
    logger.debug(f'Arguments: {args}')
    main(logger=logger, args=args)
    logger.info('Demo done.')
