"""A simple, multilayer feed forward model with categorical outputs."""

from torch_playground.util import BaseArguments, App, save_tensor, get_default_working_dir
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
import json
from collections import OrderedDict


class HRLinearMultilayer(nn.Module):
    """A simple linear classifier, without bias; trainable.

    This model assumes that data points are rows in the input tensor.
    """

    def __init__(self,
                 input_dim: int,
                 n_categories: int = 5,
                 n_hidden_layers: int = 2,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        assert input_dim > 0, 'Input dimension must be positive.'
        # Not an optimal arrangement - just a simple one for demo purposes.
        hidden_layer_widths = torch.round(torch.linspace(input_dim, n_categories, n_hidden_layers))
        layers = []
        last_layer_width = input_dim
        for w in hidden_layer_widths[1:]:
            width = int(w.item())
            layers.append(torch.nn.Linear(last_layer_width, width, dtype=dtype))
            last_layer_width = width
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(num_features=width, dtype=dtype))
        # Final layer: softmax for Pr[class]
        layers.append(torch.nn.Softmax())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layer."""
        return self.layers(x)

class DataGenerator:
    def __init__(self, dim: int, n_classes: int, dtype: torch.dtype) -> None:
        """Set up the hyperparameters for the data generation process.

        This initializes the following internal hyperparameters:

        means: [n_classes, n_dim] - one n_dim-dimensional row vector for each class,
            stacked by rows.
        covariances: [n_classes, n_dim, n_dim] - one (n_dim x n_dim) square, positive
            definite (PD) covariance matrix foe each class, stacked by rows.
        _cov_factors_cache: [n_classes, n_dim, n_dim] - one (n_dim x n_dim) square matrix
            containing the Cholesky lower triangular factor matrix of the corresponsing
            covariance, for each row.
        """
        self.dim = dim
        self.n_classes = n_classes
        S = torch.randn(size=(dim, dim), dtype=dtype)
        S = S.T @ S
        L = torch.linalg.cholesky(S)
        pd_mat_generator = torch.distributions.Wishart(df=torch.as_tensor(dim),
                                                       scale_tril=L)
        mean_generators = torch.distributions.Normal(loc=torch.zeros((dim, )), scale=torch.ones((dim, )))
        self.means = torch.stack([mean_generators.sample() for _ in range(n_classes)])
        self.covariances = torch.stack([pd_mat_generator.sample() + torch.eye(dim) * 1e-3 for _ in range(n_classes)])
        # Cache the Cholesky factors (intuitively, the "square roots") of the
        # covariance matrices. We'll need these in the generation steps.
        self._cov_factors_cache: torch.Tensor = torch.linalg.cholesky(self.covariances)

    def generate(self, n_points: int) -> TensorDataset:
        assert n_points > 0
        y = torch.randint(0, self.n_classes, size=(n_points,))
        # Start with IID normal variates.
        raw_points = torch.randn(size=(n_points, self.dim))
        # Project each row through the Cholesky decomposition of the covariance matrix
        # for its corresponding class. See
        #   https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
        # for a discussion of this approach to generating multivariate normals.
        X = torch.empty((n_points, self.dim))
        # Note: There may be a smarter way to do this that works purely in tensor operations, without
        # an explicit loops over classes. But my initial attempts at it failed. The
        # naive,
        #   X = (raw_points @ self._cov_factors_cache[y]) + self.means[y]
        # doesn't do the right things - it expands the output dimension per broadcast
        # semantics.
        for c in range(self.n_classes):
            class_indices = (y == c)
            X[class_indices, :] = (raw_points[class_indices, :] @ self._cov_factors_cache[c]) + self.means[c]
        return TensorDataset(X, y)


@dataclass
class MultilayerArguments(BaseArguments):
    """Command line arguments for the linear trainable application."""
    dim: int = field(default=10, metadata=BaseArguments._meta(help='The dimension of the input features.'))
    n_classes: int = field(default=5, metadata=BaseArguments._meta(help='Number of output classes'))
    n_hidden_layers: int = field(default=3, metadata=BaseArguments._meta(help='Number of hidden layers'))
    n_train_samples: int = field(default=100, metadata=BaseArguments._meta(help='Number of training samples to generate.'))
    epochs: int = field(default=10, metadata=BaseArguments._meta(help='Number of epochs to train the model.'))
    batch_size: int = field(default=8, metadata=BaseArguments._meta(help='Batch size for training.'))
    learning_rate: float = field(default=0.01, metadata=BaseArguments._meta(help='Learning rate for the optimizer.'))
    output_dir: Path = field(default=get_default_working_dir(),
                             metadata=BaseArguments._meta(help='Directory to save the data, checkpoints, trained model, etc.'))

class LinearTrainableApp(App[MultilayerArguments, HRLinearMultilayer]):
    """An application that trains a simple linear model."""

    def __init__(self, argv: Optional[list[str]] = None):
        super().__init__(MultilayerArguments(),
                         'Train a simple linear model to fit a separable problem in low-dimensional space.',
                         argv=argv)
        self.model: Optional[HRLinearMultilayer] = None

    def create_data(self) -> tuple[TensorDataset, torch.Tensor]:
        """Create a dataset of a d-dimensional discriminator, random inputs, and classified outputs.

        Data points are rows.

        Returns:
            TensorDataset: A dataset containing random integer inputs.
        """
        assert self.config.dim > 0, 'X dimension must be positive.'
        assert self.config.n_classes > 0, 'Number of output classes must be positive.'
        assert self.config.n_train_samples > 0, 'Number of samples must be positive.'
        self.logger.info('Creating data set',
                         dim=self.config.dim,
                         n_classes=self.config.n_classes,
                         num_train_samples=self.config.n_train_samples)
        # A zero-mean, unit-variance, normally distributed data set.
        X = torch.randn(size=(self.config.n_train_samples, self.config.dim), dtype=self.dtype).to(self.device)
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
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config.output_dir / 'args.txt', 'w') as f:
                json.dump(asdict(self.config), f, indent=2, default=str)
            data, discriminator = self.create_data()
            save_tensor(discriminator, self.config.output_dir / 'discriminator')
            save_tensor(data.tensors[0], self.config.output_dir / 'X')
            save_tensor(data.tensors[1], self.config.output_dir / 'y')
            self.model = HRLinearMultilayer(input_dim=self.config.dim,
                                            n_categories=self.config.n_classes,
                                            n_hidden_layers=self.config.n_hidden_layers,
                                            dtype=self.dtype).to(self.device)
            self.logger.info('Sample output', sample_output=self.model(data.tensors[0].to(self.device))[:5])
            self.logger.info('Model summary', model=summary(self.model, input_size=(self.config.n_train_samples, self.config.dim), verbose=0))
            self.tb_writer.add_graph(self.model, data.tensors[0])
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
            loss_fn = nn.CrossEntropyLoss()
            data_loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=True)
            self.train_model(data=data_loader,
                             optimizer=optimizer,
                             loss_fn=loss_fn,
                             num_epochs=self.config.epochs)
            with (self.config.output_dir / 'trained_model.pt').open('wb') as f:
                torch.save(self.model, f)
            self.logger.info('Model trained and saved', model_path=self.config.output_dir / 'trained_model.pt')
            self.logger.info('||target - model.W||', norm=torch.linalg.vector_norm(discriminator - self.model.W).item())
            predictions = self.classify_data(data)
            predictions = torch.stack((predictions, data.tensors[1]), dim=1)  # Stack predictions with true labels for output.
            save_tensor(predictions, self.config.output_dir / 'predictions_truth')
            self.logger.info('Predictions saved', predictions_path=self.config.output_dir / 'predictions_truth.pt')
            self.logger.info('Application run completed successfully')
        except Exception as e:
            self.logger.exception('Uncaught error somewhere in the code (hopeless).', exc_info=e)
            raise


if __name__ == '__main__':
    LinearTrainableApp().run()
