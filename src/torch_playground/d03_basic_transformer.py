"""A simple, few layer transformer model."""

from torch_playground.util import (
    BaseConfiguration,
    App,
    save_tensor,
)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
import json


class HRBasicTransformer(nn.Module):
    """A basic demo of using a Transformer.

    This uses the simplest, out of the box, Transformer demo object -- torch.nn.Transformer.
    As the docs point out, this is _not_ a modern, peak-performance xformer. It's just a
    straightforward implementation of the "Attention is All You Need" architecture. It exists
    precisely for this purpose - for hands-on learning.

    Note: We use the "batch first" data convention here, so source and target tensors must
    be of shape (batch_size, seq_length, embedded_space_dim).
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 n_encoder_layers: int,
                 n_decoder_layers: int,
                 d_feedforward: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.xformer = torch.nn.Transformer(
                d_model=d_model,
                num_encoder_layers=n_encoder_layers,
                num_decoder_layers=n_decoder_layers,
                nhead=n_heads,
                dim_feedforward=d_feedforward,
                batch_first=True,
                dtype=dtype)

    def forward(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer stack."""
        # Todo: set up masks.
        return self.xformer.forward(src=src, tgt=target)



@dataclass
class BasicTransformerConfig(BaseConfiguration):
    """Command line arguments for the simple Transformer."""
    learning_rate: float = field(default=0.01, metadata=BaseConfiguration._meta(help='Learning rate for the optimizer.'))
    d_model: int = field(default=128, metadata=BaseConfiguration._meta(help='Model internal embedding dimension.'))
    n_heads: int = field(default=4, metadata=BaseConfiguration._meta(help='Number of attentional heads.'))
    n_encoder_layers: int = field(default=3, metadata=BaseConfiguration._meta(help='Number of encoder layers.'))
    n_decoder_layers: int = field(default=3, metadata=BaseConfiguration._meta(help='Number of decoder layers.'))
    d_feedfoward: int = field(default=1024, metadata=BaseConfiguration._meta(help='Dimension of the final dense feedforward layer.'))


class BasicTransformerApp(App[BasicTransformerConfig, HRBasicTransformer]):
    """An application that trains a simple linear model."""

    def __init__(self, argv: Optional[list[str]] = None):
        super().__init__(BasicTransformerConfig(),
                         'Train a small, basic Transformer model.',
                         argv=argv)
        self.model: Optional[HRBasicTransformer] = None

    def run(self):
        # TODO(heather): This is common boilerplate, should be moved to App.
        try:
            self.logger.info('Starting BasicTranformer demo app with arguments', **asdict(self.config))
            self.model = HRBasicTransformer(d_model=self.config.d_model,
                                            n_heads=self.config.n_heads,
                                            n_encoder_layers=self.config.n_encoder_layers,
                                            n_decoder_layers=self.config.n_decoder_layers,
                                            d_feedforward=self.config.d_feedfoward,
                                            dtype=self.dtype).to(self.device)
            self.model.eval()  # We're not training for the moment.
            placeholder_src = torch.ones(self.config.batch_size, 20, self.config.d_model)  # batch_size items of len 20 and dim d_model.
            placeholder_target = torch.ones(self.config.batch_size, 10, self.config.d_model)  # batch size items of len 10 and dim d_model.
            model_summary = summary(self.model, input_size=(placeholder_src.shape, placeholder_target.shape), verbose=0)
            self.logger.info('Model summary', model=model_summary)
            (self.work_dir / 'model_summary.txt').write_text(str(model_summary))
            return
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
    BasicTransformerApp().run()
