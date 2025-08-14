"""A simple, few layer transformer model."""

from torch_playground.util import (
    BaseArguments,
    App,
    save_tensor,
    get_default_working_dir,
)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
import json
import structlog


class HRBasicTransformer(nn.Module):
    """A simple linear classifier, without bias; trainable.

    This model assumes that data points are rows in the input tensor.
    """

    def __init__(self,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layer."""
        pass



@dataclass
class BasicTransformerConfig(BaseArguments):
    """Command line arguments for the simple Transformer."""
    epochs: int = field(default=10, metadata=BaseArguments._meta(help='Number of epochs to train the model.'))
    batch_size: int = field(default=8, metadata=BaseArguments._meta(help='Batch size for training.'))
    learning_rate: float = field(default=0.01, metadata=BaseArguments._meta(help='Learning rate for the optimizer.'))
    output_dir: Path = field(default=get_default_working_dir(),
                             metadata=BaseArguments._meta(help='Directory to save the data, checkpoints, trained model, etc.'))

class LinearTrainableApp(App[BasicTransformerConfig, HRBasicTransformer]):
    """An application that trains a simple linear model."""

    def __init__(self, argv: Optional[list[str]] = None):
        super().__init__(BasicTransformerConfig(),
                         'Train a simple linear model to fit a separable problem in low-dimensional space.',
                         argv=argv)
        self.model: Optional[HRBasicTransformer] = None

    def run(self):
        # TODO(heather): This is common boilerplate, should be moved to App.
        try:
            self.logger.info('Starting LinearTrainableApp with arguments', **asdict(self.config))
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.config.output_dir / 'config.json').open('wt') as f:
                json.dump(asdict(self.config), f, indent=2, default=str)
            self.model = HRBasicTransformer(dtype=self.dtype).to(self.device)
            self.model.eval()  # We're not training for the moment.
            self.logger.info('Sample output', sample_output=self.model(data.tensors[0].to(self.device))[:5])
            self.logger.info('Model summary', model=summary(self.model, input_size=(self.config.n_train_samples, self.config.dim), verbose=0))
            # self.tb_writer.add_graph(self.model, data.tensors[0])
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
            loss_fn = nn.NLLLoss()
            data_loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=True)
            self.train_model(data=data_loader,
                             optimizer=optimizer,
                             loss_fn=loss_fn,
                             num_epochs=self.config.epochs)
            with (self.config.output_dir / 'trained_model.pt').open('wb') as f:
                torch.save(self.model, f)
            self.logger.info('Model trained and saved', model_path=self.config.output_dir / 'trained_model.pt')
            self.logger.info('Application run completed successfully')
        except Exception as e:
            self.logger.exception('Uncaught error somewhere in the code (hopeless).', exc_info=e)
            raise


if __name__ == '__main__':
    LinearTrainableApp().run()
