"""A sequence learner for human names (or other short text fragments)."""

from torch_playground.util import (
    BaseConfiguration,
    TrainableModelApp,
    save_tensor,
    SequenceCrossEntropyLoss,
    InMemoryFileDataset,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchinfo import summary
from typing import Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from torch_playground.tokenizer import NGramTokenizer


@dataclass
class NameSeqLearnerConfig(BaseConfiguration):
    names_file: Path = field(default=Path('/dev/null'),
                             metadata=BaseConfiguration._meta(help='File name containing newline-separated name records.',
                                                              required=True))
    d_embedding: int = field(default=128,
                             metadata=BaseConfiguration._meta(help='Dimension of embedding space.'))
    n_heads: int = field(default=4, metadata=BaseConfiguration._meta(help='Number of attentional heads. Must divide d_model exactly.'))
    n_encoder_layers: int = field(default=3, metadata=BaseConfiguration._meta(help='Number of encoder layers.'))
    n_decoder_layers: int = field(default=3, metadata=BaseConfiguration._meta(help='Number of decoder layers.'))
    d_feedforward: int = field(default=1024, metadata=BaseConfiguration._meta(help='Dimension of the final dense feedforward layer.'))
    in_seq_length: int = field(default=64, metadata=BaseConfiguration._meta('Length of input sequences (context window).'))
    out_seq_length: int = field(default=64, metadata=BaseConfiguration._meta('Length of output sequences (response length).'))
    vocab_size: int = field(default=0, metadata=BaseConfiguration._meta('Size of the vocabulary (number of unique tokens). Overwritten by tokenizer vocab size at runtime.'))
    tokenizer_file: Path = field(default=Path('/dev/null'),
                                 metadata=BaseConfiguration._meta(help='JSON file containing tokenizer state.'))
    learning_rate: float = field(default=0.01,
                                 metadata=BaseConfiguration._meta(help='SGD learning rate.'))


class NameSeqTransformer(nn.Module):
    def __init__(self,
                 d_embedding: int,
                 n_heads: int,
                 n_encoder_layers: int,
                 n_decoder_layers: int,
                 d_feedforward: int,
                 vocab_size: int,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=d_embedding,
                                            max_norm=1.0,
                                            dtype=dtype)
        self.xformer = torch.nn.Transformer(
            d_model=d_embedding,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            batch_first=True,
            dtype=dtype)
        self.logits_layer = torch.nn.Linear(d_embedding, vocab_size, dtype=dtype)

    @staticmethod
    def from_config(config: NameSeqLearnerConfig, dtype: torch.dtype = torch.float32) -> 'NameSeqTransformer':
        return NameSeqTransformer(d_embedding=config.d_embedding,
                                  n_heads=config.n_heads,
                                  n_encoder_layers=config.n_encoder_layers,
                                  n_decoder_layers=config.n_decoder_layers,
                                  d_feedforward=config.d_feedforward,
                                  vocab_size=config.vocab_size,
                                  dtype=dtype)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        xformer_out = self.xformer(src=src_emb, tgt=tgt_emb)
        logits = self.logits_layer(xformer_out)
        return logits


class PaddingCollate:
    """Collate a list of token ID lists into a single batch of tensors, including padding.

    Args:
        padding_value (int): Value to use for padding shorter sequences.
    """
    def __init__(self, padding_value: int):
        self.padding_value = padding_value

    def __call__(self, batch: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor_batch = [torch.tensor(item, dtype=torch.long) for item in batch]
        padded_batch = pad_sequence(tensor_batch, batch_first=True, padding_value=self.padding_value)
        # TODO(hlane): Return attention masks as well.
        # TODO(hlane): Shift target to right by one position.
        return padded_batch, padded_batch, padded_batch  # src, tgt, target


class NameSeqLearnerApp(TrainableModelApp[NameSeqLearnerConfig, NameSeqTransformer]):
    def __init__(self, argv: Optional[list[str]] = None):
        super().__init__(NameSeqLearnerConfig(),
                         'Train a Transformer model to generate human names.',
                         argv=argv)
        self.tokenizer = NGramTokenizer.from_file(self.config.tokenizer_file)
        self.logger.info('Loaded tokenizer', n_tokens=self.tokenizer.vocab_size())
        self.config.vocab_size = self.tokenizer.vocab_size()
        self.model: NameSeqTransformer | None = None

    def run(self):
        try:
            self.logger.info('Starting NameSeqLearner app with arguments', **asdict(self.config))
            self.model = NameSeqTransformer.from_config(self.config, dtype=self.dtype).to(self.device)
            self.model.eval()  # We're not training for the moment.
            full_data = InMemoryFileDataset(self.config.names_file).with_transform(str.strip).with_transform(self.tokenizer.tokenize)
            train, test, val = random_split(dataset=full_data, lengths=[0.7, 0.15, 0.15])  # type: ignore
            self.logger.info('Split full data', train_size=len(train), test_size=len(test), val_size=len(val))
            for d_part, name in [(train, 'train'), (test, 'test'), (val, 'val')]:
                (self.work_dir / f'{name}_indices.txt').write_text('\n'.join([str(x) for x in d_part.indices]))
            model_summary = summary(self.model, verbose=0)
            self.logger.info('Model summary', model=model_summary)
            (self.work_dir / 'model_summary.txt').write_text(str(model_summary))
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
            loss_fn = SequenceCrossEntropyLoss()
            pad_value = self.tokenizer.get_token_id(self.tokenizer.padding_token)  # type: ignore
            if pad_value is None:
                pad_value = 0
            train_loader = DataLoader(train,
                                      batch_size=self.config.batch_size,
                                      shuffle=True,
                                      collate_fn=PaddingCollate(padding_value=pad_value))
            # TODO(hlane) Add support for holdout test/val data.
            self.train_model(data=train_loader,
                             optimizer=optimizer,
                             loss_fn=loss_fn)
            with (self.work_dir / 'trained_model.pt').open('wb') as f:
                torch.save(self.model, f)
            self.logger.info('Model trained and saved', model_path=self.work_dir / 'trained_model.pt')
            self.logger.info('Application run completed successfully')
        except Exception as e:
            torch.set_printoptions(precision=2, threshold=7, edgeitems=2, linewidth=60)
            self.logger.exception('Uncaught error somewhere in the code (hopeless).', exc_info=e)
            raise


if __name__ == '__main__':
      NameSeqLearnerApp().run()