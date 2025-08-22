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
from dataclasses import dataclass, field, asdict
import math


class HRLBasicTransformer(nn.Module):
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
                 vocab_size: int,
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
        # For vocab_size > d_model, it's not possible to exactly pick an orthogonal vector set. But
        # we can get asymptotically "approximately orthogonal" by simply picking normally distributed
        # random vectors.
        # Shape: (vocab_sizw, d_model) so that embedding_mapping[k, :] is the embedding of the k'th symbol
        # in the vocabulary.
        self.embedding_mapping = torch.nn.functional.normalize(torch.randn((vocab_size, d_model)), dim=1)

    def embed(self, src: torch.Tensor) -> torch.Tensor:
        """Embed the source tensor.

        Embedding is a mapping from, in this case, Z^n -> R^d*n, for a sequence of n integers
        to a sequence of n vectors of dimension d. Here we're using a simple Fourier orthogonal
        basis for demo purposes.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, seq_length) and an int dtype.

        Returns:
            torch.Tensor: Embedded source tensor of shape (batch_size, seq_length, d_model),
            matching the expected input of the transformer.
        """
        return self.embedding_mapping[src]

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
    in_seq_length: int = field(default=64, metadata=BaseConfiguration._meta('Length of input sequences (context window).'))
    out_seq_length: int = field(default=32, metadata=BaseConfiguration._meta('Length of output sequences (response length).'))
    vocab_size: int = field(default=2048, metadata=BaseConfiguration._meta('Size of the vocabulary (number of unique tokens).'))


def sieve(n: int) -> list[bool]:
    """Generate list of primes via the Sieve of Eratosthenes.

    Args:
        n (int): Largest number to search up to (inclusive).

    Returns:
        list[bool]: Mapping from int to bool: l[i] == True iff i is prime. Len(l) == n + 1
            (since both 0 and n are included).
    """
    assert n >= 0
    result: list[bool] = [True] * (n + 1)
    result[0] = False
    if n == 0:
        return result
    result[1] = False
    curr_prime = 2
    upper_bound = math.ceil(math.sqrt(n))
    try:
        while curr_prime <= upper_bound:
            for i in range(2 * curr_prime, n + 1, curr_prime):
                result[i] = False
            curr_prime = result.index(True, curr_prime + 1)
    except ValueError:
        # No more primes left in list - terminate sieve.
        pass
    return result


def generate_data(n_points: int, in_seq_length: int, out_seq_length: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Toy data generator: Let's try to learn the notion of primality."""
    # We want a Toeplitz matrix with ascending integer sequences as rows. This is not
    # a super efficient way to construct it, but as long as the data is small. :shrug:
    input_int_seqs = torch.empty((n_points, in_seq_length))
    output_int_seqs = torch.empty((n_points, out_seq_length))
    primes = torch.as_tensor(sieve(n_points + in_seq_length))
    for i in range(n_points):
        input_int_seqs[i, :] = torch.as_tensor(range(i, i + in_seq_length))
    return input_int_seqs, output_int_seqs


class BasicTransformerApp(App[BasicTransformerConfig, HRLBasicTransformer]):
    """An application that trains a simple linear model."""

    def __init__(self, argv: Optional[list[str]] = None):
        super().__init__(BasicTransformerConfig(),
                         'Train a small, basic Transformer model.',
                         argv=argv)
        self.model: Optional[HRLBasicTransformer] = None

    def run(self):
        # TODO(heather): This is common boilerplate, should be moved to App.
        try:
            self.logger.info('Starting BasicTranformer demo app with arguments', **asdict(self.config))
            self.model = HRLBasicTransformer(d_model=self.config.d_model,
                                            n_heads=self.config.n_heads,
                                            n_encoder_layers=self.config.n_encoder_layers,
                                            n_decoder_layers=self.config.n_decoder_layers,
                                            d_feedforward=self.config.d_feedfoward,
                                            dtype=self.dtype).to(self.device)
            self.model.eval()  # We're not training for the moment.
            placeholder_src = torch.ones(self.config.batch_size, self.config.in_seq_length, self.config.d_model)  # batch_size items of len in_seq_len and dim d_model.
            placeholder_target = torch.ones(self.config.batch_size, self.config.out_seq_length, self.config.d_model)  # batch size items of len out_seq_len and dim d_model.
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
