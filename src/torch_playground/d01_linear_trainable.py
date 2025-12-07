"""The most basic trainable model: single linear transform, no bias."""

from torch_playground.util import BaseConfiguration, TrainableModelApp, save_tensor, get_default_working_dir
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict


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
        return torch.sign(x @ self.W)


@dataclass
class LinearTrainableArguments(BaseConfiguration):
    """Command line arguments for the linear trainable application."""
    dim: int = field(default=10, metadata=BaseConfiguration._meta(help='The dimension of the input features.'))
    num_train_samples: int = field(default=100, metadata=BaseConfiguration._meta(help='Number of training samples to generate.'))
    learning_rate: float = field(default=0.01, metadata=BaseConfiguration._meta(help='Learning rate for the optimizer.'))

class LinearTrainableApp(TrainableModelApp[LinearTrainableArguments, HRLinearTrainable]):
    """An application that trains a simple linear model."""

    def __init__(self, argv: Optional[list[str]] = None):
        super().__init__(LinearTrainableArguments(),
                         'Train a simple linear model to fit a separable problem in low-dimensional space.',
                         argv=argv)
        self.model: Optional[HRLinearTrainable] = None

    def create_data(self) -> tuple[TensorDataset, torch.Tensor]:
        """Create a dataset of a d-dimensional discriminator, random inputs, and classified outputs.

        Data points are rows.

        Returns:
            TensorDataset: A dataset containing random integer inputs.
        """
        assert self.config.dim > 0, 'X dimension must be positive.'
        assert self.config.num_train_samples > 0, 'Number of samples must be positive.'
        self.logger.info('Creating data set', dim=self.config.dim, num_train_samples=self.config.num_train_samples)
        # A zero-mean, unit-variance, normally distributed data set.
        X = torch.randn(size=(self.config.num_train_samples, self.config.dim), dtype=self.dtype).to(self.device)
        self.logger.debug('X[0:3]', sample=X[:3])
        discriminator = torch.randn(size=(self.config.dim,), dtype=self.dtype)
        self.logger.debug('Discriminator', sample=discriminator)
        # Classify the data points based on the discriminator.
        y = torch.sign(X @ discriminator).to(self.device)
        self.logger.debug('y[0:5]', sample=y[:5])
        self.logger.info('Data set created',
                         num_samples=X.shape[0],
                         dim=X.shape[1],
                         num_positive_samples=(y > 0).sum().item(),
                         num_negative_samples=(y < 0).sum().item())
        return TensorDataset(X, y), discriminator

    def classify_data(self, data: TensorDataset) -> torch.Tensor:
        """Classify the data using the trained model.

        Args:
            data (TensorDataset): The dataset containing input features.

        Returns:
            torch.Tensor: The model's predictions for the input data.
        """
        assert self.model is not None, 'Model must be initialized before classification.'
        self.logger.info('Classifying data with the trained model')
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            X = data.tensors[0].to(self.device)  # Move input data to the appropriate device
            predictions = self.model(X)
        return predictions

    def run(self):
        # TODO(heather): This is common boilerplate, should be moved to App.
        try:
            self.logger.info('Starting LinearTrainableApp with arguments', **asdict(self.config))
            self.work_dir.mkdir(parents=True, exist_ok=True)
            data, discriminator = self.create_data()
            save_tensor(discriminator, self.work_dir / 'discriminator')
            save_tensor(data.tensors[0], self.work_dir / 'X')
            save_tensor(data.tensors[1], self.work_dir / 'y')
            self.model = HRLinearTrainable(input_dim=self.config.dim, dtype=self.dtype).to(self.device)
            self.logger.info('Sample output', sample_output=self.model(data.tensors[0].to(self.device))[:5])
            self.logger.info('Model summary', model=summary(self.model, input_size=(self.config.num_train_samples, self.config.dim), verbose=0))
            self.tb_writer.add_graph(self.model, data.tensors[0])
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
            loss_fn = nn.MSELoss(reduction='mean')
            data_loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=True)
            self.train_model(train_data=data_loader,
                             optimizer=optimizer,
                             loss_fn=loss_fn)
            save_tensor(self.model.W, self.work_dir / 'W_trained')
            self.logger.info('Model trained and saved', model_path=self.work_dir / 'W_trained.pt')
            self.logger.info('||target - model.W||', norm=torch.linalg.vector_norm(discriminator - self.model.W).item())
            predictions = self.classify_data(data)
            predictions = torch.stack((predictions, data.tensors[1]), dim=1)  # Stack predictions with true labels for output.
            save_tensor(predictions, self.work_dir / 'predictions_truth')
            self.logger.info('Predictions saved', predictions_path=self.work_dir / 'predictions_truth.pt')
            self.logger.info('Application run completed successfully')
        except Exception as e:
            self.logger.exception('Uncaught error somewhere in the code (hopeless).', exc_info=e)
            raise


if __name__ == '__main__':
    LinearTrainableApp().run()
