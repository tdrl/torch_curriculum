"""Simplest possible torch demo: linear projection with integer values."""

from torch_playground.util import setup_logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
from typing import Optional
from pathlib import Path
from logging import Logger
from dataclasses import dataclass


class HRLinear(nn.Module):
    """A simple linear projection model."""

    def __init__(self, input_dim: int, output_dim: int, W: Optional[torch.Tensor] = None, b: Optional[torch.Tensor] = None):
        super(HRLinear, self).__init__()
        if W is None:
            self.W = torch.randint(-10, 10, (input_dim, output_dim), dtype=torch.int32)
        else:
            self.W = W
        if b is None:
            self.b = torch.randint(-10, 10, (output_dim,), dtype=torch.int32)
        else:
            self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layer."""
        return x @ self.W + self.b


def create_data(input_dim: int, n: int) -> TensorDataset:
    """Create a dataset of random integer inputs and outputs."""
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
    output_dir = Path(args.output_dir) / '00_linear_projection'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Output directory: {output_dir}')
    model = HRLinear(input_dim=args.input_dim, output_dim=args.output_dim)
    torch.save(model.state_dict(), output_dir / 'model.pth')
    with open(output_dir / 'model.txt', 'w') as f:
        f.write(str(model))
    logger.debug(f'Model parameters: W={model.W}, b={model.b}')
    # model.compile()
    x = create_data(args.input_dim, args.n_samples)
    torch.save(x, output_dir / 'x.pt')
    with open(output_dir / 'x.txt', 'w') as f:
        f.write(str(x.tensors[0].tolist()))
    y = model(x)
    torch.save(y, output_dir / 'y.pt')
    for i in range(args.n_samples):
        logger.info(f'{i}: {x[i]} -> {i}: {y[i]}')


if __name__ == '__main__':
    logger = setup_logging()
    logger.info('Starting linear projection demo.')
    args = parse_args()
    logger.debug(f'Arguments: {args}')
    main(logger=logger, args=args)
    logger.info('Demo done.')
