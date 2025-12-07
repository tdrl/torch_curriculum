"""A simple, multilayer feed forward model with categorical outputs."""

from torch_playground.util import (
    BaseConfiguration,
    TrainableModelApp,
    save_tensor,
    get_default_working_dir,
    accuracy
)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
import structlog


class HRLinearMultilayer(nn.Module):
    """A simple linear classifier, without bias; trainable.

    This model assumes that data points are rows in the input tensor.
    """

    def __init__(self,
                 input_dim: int,
                 n_classes: int = 5,
                 n_hidden_layers: int = 2,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        assert input_dim > 0, 'Input dimension must be positive.'
        # Not an optimal arrangement - just a simple one for demo purposes.
        hidden_layer_widths = torch.round(torch.linspace(input_dim, n_classes, n_hidden_layers))
        layers = []
        last_layer_width = input_dim
        for w in hidden_layer_widths[1:]:
            width = int(w.item())
            layers.append(torch.nn.Linear(last_layer_width, width, dtype=dtype))
            last_layer_width = width
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(num_features=width, dtype=dtype))
        # Final layer: softmax for Pr[class]
        layers.append(torch.nn.Softmax(dim=0))
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
        logger = structlog.get_logger()
        self.dim = dim
        self.n_classes = n_classes
        S = torch.randn(size=(dim, dim), dtype=dtype)
        S = S.T @ S
        L = torch.linalg.cholesky(S)
        pd_mat_generator = torch.distributions.Wishart(df=torch.as_tensor(dim),
                                                       scale_tril=L)
        mean_generators = torch.distributions.Normal(loc=torch.zeros((dim, )), scale=10 * torch.ones((dim, )))
        # TODO(heather): Scale the mean norms more intelligently by inspecting the
        # covariances (ideally: look at the max eigenvalue over all cov matrices).
        self.means = torch.stack([mean_generators.sample() for _ in range(n_classes)])
        self.covariances = torch.stack([pd_mat_generator.sample() + torch.eye(dim) * 1e-3 for _ in range(n_classes)])
        # Cache the Cholesky factors (intuitively, the "square roots") of the
        # covariance matrices. We'll need these in the generation steps.
        self._cov_factors_cache: torch.Tensor = torch.linalg.cholesky(self.covariances)
        logger.debug('Data generator initialized',
                     means_dim=self.means.shape,
                     cov_dim=self.covariances.shape,
                     mean_norms=torch.linalg.vector_norm(self.means, dim=1))

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
class MultilayerArguments(BaseConfiguration):
    """Command line arguments for the linear trainable application."""
    dim: int = field(default=10, metadata=BaseConfiguration._meta(help='The dimension of the input features.'))
    n_classes: int = field(default=5, metadata=BaseConfiguration._meta(help='Number of output classes'))
    n_hidden_layers: int = field(default=3, metadata=BaseConfiguration._meta(help='Number of hidden layers'))
    n_train_samples: int = field(default=100, metadata=BaseConfiguration._meta(help='Number of training samples to generate.'))
    n_val_samples: int = field(default=100, metadata=BaseConfiguration._meta(help='Number of holdout validation set samples to generate.'))
    learning_rate: float = field(default=0.01, metadata=BaseConfiguration._meta(help='Learning rate for the optimizer.'))


class LinearTrainableApp(TrainableModelApp[MultilayerArguments, HRLinearMultilayer]):
    """An application that trains a simple linear model."""

    def __init__(self, argv: Optional[list[str]] = None):
        super().__init__(MultilayerArguments(),
                         'Train a simple linear model to fit a separable problem in low-dimensional space.',
                         argv=argv)
        self.model: Optional[HRLinearMultilayer] = None

    def hard_classify_data(self, data: TensorDataset) -> torch.Tensor:
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
            predictions = torch.argmax(self.model(X), dim=1)
        return predictions

    def run(self):
        # TODO(heather): This is common boilerplate, should be moved to App.
        try:
            self.logger.info('Starting LinearTrainableApp with arguments', **asdict(self.config))
            data_generator = DataGenerator(dim=self.config.dim, n_classes=self.config.n_classes, dtype=self.dtype)
            data = data_generator.generate(n_points=self.config.n_train_samples)
            self.logger.debug('Synthesized data',
                              n_points=len(data),
                              dim=data.tensors[0].shape[1],
                              n_classes=torch.unique(data.tensors[1]).shape[0])
            save_tensor(data_generator.covariances, self.work_dir / 'hyper_cov')
            save_tensor(data_generator.means, self.work_dir / 'hyper_means')
            save_tensor(data.tensors[0], self.work_dir / 'X')
            save_tensor(data.tensors[1], self.work_dir / 'y')
            self.tb_writer.add_embedding(data_generator.means,
                                         metadata=torch.arange(0, self.config.n_classes),
                                         global_step=0,
                                         tag='Cluster mean vectors')
            display_count = min(30 * self.config.n_classes, self.config.n_train_samples)
            self.logger.debug('Writing tensorboard embedding sample', count=display_count)
            self.tb_writer.add_embedding(data.tensors[0][:display_count],
                                         metadata=data.tensors[1][:display_count],
                                         global_step=0,
                                         tag='Data class distribution')
            self.model = HRLinearMultilayer(input_dim=self.config.dim,
                                            n_classes=self.config.n_classes,
                                            n_hidden_layers=self.config.n_hidden_layers,
                                            dtype=self.dtype).to(self.device)
            self.model.eval()  # We're not training for the moment.
            self.logger.info('Sample output', sample_output=self.model(data.tensors[0].to(self.device))[:5])
            self.logger.info('Model summary', model=summary(self.model, input_size=(self.config.n_train_samples, self.config.dim), verbose=0))
            self.tb_writer.add_graph(self.model, data.tensors[0])
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            loss_fn = nn.NLLLoss()
            data_loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=True)
            self.train_model(train_data=data_loader,
                             optimizer=optimizer,
                             loss_fn=loss_fn)
            with (self.work_dir / 'trained_model.pt').open('wb') as f:
                torch.save(self.model, f)
            self.logger.info('Model trained and saved', model_path=self.work_dir / 'trained_model.pt')
            predictions = self.hard_classify_data(data)
            predictions = torch.stack((predictions, data.tensors[1]), dim=1)  # Stack predictions with true labels for output.
            save_tensor(predictions, self.work_dir / 'predictions_truth')
            self.logger.info('Predictions saved', predictions_path=self.work_dir / 'predictions_truth.pt')
            self.logger.info('Final train-set accuracy', accuracy=accuracy(predictions[:, 0], predictions[:, 1]))
            validation_data = data_generator.generate(n_points=self.config.n_val_samples)
            val_predictions = self.hard_classify_data(validation_data)
            save_tensor(torch.stack((val_predictions, validation_data.tensors[1])), self.work_dir / 'val_predictions_truth')
            self.logger.info('Final validation-set accuracy', accuracy=accuracy(val_predictions, validation_data.tensors[1]))
            self.logger.info('Application run completed successfully')
        except Exception as e:
            self.logger.exception('Uncaught error somewhere in the code (hopeless).', exc_info=e)
            raise


if __name__ == '__main__':
    LinearTrainableApp().run()
