"""A simple, few layer transformer model."""

from torch_playground.util import (
    BaseConfiguration,
    TrainableModelApp,
    save_tensor,
    SequenceCrossEntropyLoss
)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch._prims_common import DeviceLikeType
from torchinfo import summary
from typing import Optional
from dataclasses import dataclass, field, asdict
import math


@dataclass
class BasicTransformerConfig(BaseConfiguration):
    """Command line arguments for the simple Transformer."""
    learning_rate: float = field(default=0.01, metadata=BaseConfiguration._meta(help='Learning rate for the optimizer.'))
    d_model: int = field(default=128, metadata=BaseConfiguration._meta(help='Model internal embedding dimension.'))
    n_heads: int = field(default=4, metadata=BaseConfiguration._meta(help='Number of attentional heads. Must divide d_model exactly.'))
    n_encoder_layers: int = field(default=3, metadata=BaseConfiguration._meta(help='Number of encoder layers.'))
    n_decoder_layers: int = field(default=3, metadata=BaseConfiguration._meta(help='Number of decoder layers.'))
    d_feedforward: int = field(default=1024, metadata=BaseConfiguration._meta(help='Dimension of the final dense feedforward layer.'))
    in_seq_length: int = field(default=64, metadata=BaseConfiguration._meta('Length of input sequences (context window).'))
    out_seq_length: int = field(default=64, metadata=BaseConfiguration._meta('Length of output sequences (response length).'))
    vocab_size: int = field(default=2048, metadata=BaseConfiguration._meta('Size of the vocabulary (number of unique tokens).'))
    n_points: int = field(default=1000, metadata=BaseConfiguration._meta(help='Number of points to synthesize for train/test/val data set.'))


class HRLBasicTransformer(nn.Module):
    """A basic demo of using a Transformer.

    This uses the simplest, out of the box, Transformer demo object -- torch.nn.Transformer.
    As the docs point out, this is _not_ a modern, peak-performance xformer. It's just a
    straightforward implementation of the "Attention is All You Need" architecture. It exists
    precisely for this purpose - for hands-on learning.

    Note: We use the "batch first" data convention here, so source and target tensors must
    be of shape (batch_size, seq_length, embedded_space_dim).

    Most of the args go direct into the torch.nn.Transformer model - see its docs for
    precise semantics.

    Args:
        d_model (int): Embedding dimension in which model is manipulating symbols.
        n_heads (int): Number of attention heads.
            Note: d_model must be a multiple of n_heads (d_model % n_heads == 0).
        n_decoder_layers (int): Number of decoder layers.
        n_encoder_layers (int): Number of encoder layers.
        d_feedforward (int): Dimension of final feedforward dense layers.
        vocab_size (int): Vocabulary size for encoding/decoding layers.
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
        self.vocab_size = vocab_size
        self.embedding_mapping = torch.nn.functional.normalize(torch.randn((vocab_size, d_model)), dim=1)
        # Decoding: We'll use cosine similarity along the embedding dimension and exploit broadcast
        # semantics to map the input (src) over the vocabulary dimension. The embedding dimension is the
        # final one, but we need to expand src by introducing a length 1 pseudo-dimension to make the
        # broadcast semantics right, so both tensors will have effective length 4. Hence, the final dimension
        # is 3.
        # Src shape: (b, s, e) => (b, s, 1, e) (via unsqueeze)
        # Embed tensor shape: (v, e) => (1, 1, v, e) (virtually via broadcast semantics)
        self.decoder = torch.nn.CosineSimilarity(dim=3)

    @staticmethod
    def from_config(config: BasicTransformerConfig, dtype: torch.dtype = torch.float32) -> 'HRLBasicTransformer':
        """Factory method: Create an instance of HRLBasicTransformer given a config object."""
        return HRLBasicTransformer(d_model=config.d_model,
                                   n_heads=config.n_heads,
                                   n_encoder_layers=config.n_encoder_layers,
                                   n_decoder_layers=config.n_encoder_layers,
                                   d_feedforward=config.d_feedforward,
                                   vocab_size=config.vocab_size,
                                   dtype=dtype)

    def to(self, device: DeviceLikeType | None = None, dtype: torch.dtype | str | None = None, # type: ignore
           non_blocking: bool = False) -> 'HRLBasicTransformer':
        """Override to method to ensure the embedding mapping is also moved to the device."""
        result = super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        result.embedding_mapping = result.embedding_mapping.to(device=device, dtype=dtype, non_blocking=non_blocking)  # type: ignore
        return result

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        """Embed a data tensor.

        Embedding is a mapping from, in this case, Z^n -> R^d*n, for a sequence of n integers
        to a sequence of n vectors of dimension d. Here we're using a simple Fourier orthogonal
        basis for demo purposes.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, seq_length) and an int dtype.

        Returns:
            torch.Tensor: Embedded source tensor of shape (batch_size, seq_length, d_model),
            matching the expected input of the transformer.
        """
        return self.embedding_mapping[data.to(dtype=torch.int32)]

    def decode(self, src: torch.Tensor) -> torch.Tensor:
        """Decode an embedded vector space into logits over symbols.

        Args:
            src (torch.Tensor): Input Tensor of shape (batch, seq, embed_dim)

        Returns
            torch.Tensor: Decoded sequences of shape (batch, seq, vocab_size), where
                out[b, s, j] is the logit of batch element b, position s in the sequence,
                likelihood of being symbol j.
        """
        return self.decoder(torch.unsqueeze(src, -2), self.embedding_mapping)

    def forward(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer stack."""
        # Dims: xformer.forward takes (batch, in_seq, embed_dim) X (batch, out_seq, embed_dim) -> (batch, out_seq, embed_dim)
        src_embedded = self.embed(data=src)
        tgt_embedded = self.embed(data=target)
        # TODO(hlane) Get masking sorted
        # TODO(hlane) We may want to bias this to something other than a pure coin flip.
        # target_mask = torch.randint_like(tgt_embedded, high=2)
        result = self.xformer(src=src_embedded, tgt=tgt_embedded)
        return self.decode(result)


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


def generate_data(n_points: int, seq_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Toy data generator: Let's try to learn the notion of primality.

    Args:
        n_points (int): Number of sequential numerical sequences to generate.
        seq_lengths (int): How long each sequential sequence is.

    Returns:
        tuple[Tensor, Tensor, Tensor]: (Tensor of numerical sequences (int type),
            Tensor of "is prime" labels (bool projected to int type), Tensor of "is prime" labels
            (bool projected to long, because CrossEntropyLoss requires categorical results to be
            long type.))
    """
    # We want a Toeplitz matrix with ascending integer sequences as rows. This is not
    # a super efficient way to construct it, but as long as the data is small. :shrug:
    input_int_seqs = torch.empty((n_points, seq_length), dtype=torch.int32)
    output_int_seqs = torch.empty((n_points, seq_length), dtype=torch.int32)
    primes = torch.as_tensor(sieve(n_points + seq_length))
    for i in range(n_points):
        input_int_seqs[i, :] = torch.arange(i, i + seq_length)
        output_int_seqs[i, :] = primes[input_int_seqs[i, :]]
    return input_int_seqs, output_int_seqs, output_int_seqs.to(dtype=torch.int64)  # Temp experiment


class BasicTransformerApp(TrainableModelApp[BasicTransformerConfig, HRLBasicTransformer]):
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
            self.model = HRLBasicTransformer.from_config(self.config, dtype=self.dtype)
            self.model.eval()  # We're not training for the moment.
            data = TensorDataset(*generate_data(self.config.n_points, self.config.in_seq_length))
            save_tensor(data.tensors[0], self.work_dir / 'in_seq_data')
            save_tensor(data.tensors[1], self.work_dir / 'out_seq_data')
            train, test, val = random_split(dataset=data, lengths=[0.7, 0.15, 0.15])
            self.logger.info('Split full data', full_data_size=len(data), train_size=len(train), test_size=len(test), val_size=len(val))
            for d_part, name in [(train, 'train'), (test, 'test'), (val, 'val')]:
                (self.work_dir / f'{name}_indices.txt').write_text('\n'.join([str(x) for x in d_part.indices]))
            model_summary = summary(self.model,
                                    input_data=(data.tensors[0][:self.config.batch_size, :],
                                                data.tensors[1][:self.config.batch_size, :]),
                                    verbose=0)
            self.logger.info('Model summary', model=model_summary)
            (self.work_dir / 'model_summary.txt').write_text(str(model_summary))
            self.model = self.model.to(self.device)
            self.logger.info('Model moved to device', device=self.device)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
            loss_fn = SequenceCrossEntropyLoss()
            train_loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True)
            test_loader = DataLoader(test, batch_size=self.config.batch_size)
            validate_loader = DataLoader(val, batch_size=self.config.batch_size)
            self.train_model(train_data=train_loader,
                             optimizer=optimizer,
                             loss_fn=loss_fn,
                             validate_data=validate_loader,
                             test_data=test_loader)
            with (self.work_dir / 'trained_model.pt').open('wb') as f:
                torch.save(self.model, f)
            self.logger.info('Model trained and saved', model_path=self.work_dir / 'trained_model.pt')
            self.logger.info('Application run completed successfully')
        except Exception as e:
            torch.set_printoptions(precision=2, threshold=7, edgeitems=2, linewidth=60)
            self.logger.exception('Uncaught error somewhere in the code (hopeless).', exc_info=e)
            raise


if __name__ == '__main__':
    BasicTransformerApp().run()
