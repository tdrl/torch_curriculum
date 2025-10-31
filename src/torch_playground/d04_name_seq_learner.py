"""A sequence learner for human names (or other short text fragments)."""

from torch_playground.util import (
    BaseConfiguration,
    TrainableModelApp,
    save_tensor,
    SequenceCrossEntropyLoss
)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchinfo import summary
from typing import Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


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
    vocab_size: int = field(default=2048, metadata=BaseConfiguration._meta('Size of the vocabulary (number of unique tokens).'))
    tokenizer_dict_file: Path = field(default=Path('/dev/null'),
                                      metadata=BaseConfiguration._meta(help='JSON file containing <token>:<id> mappings.'))


class NameSeqTransformer(nn.Module):
    PADDING_INDEX: int = 0
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
        embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=d_embedding,
                                       max_norm=1.0,
                                       padding_idx=self.PADDING_INDEX)
        xformer = torch.nn.Transformer(
            d_model=d_embedding,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            batch_first=True,
            dtype=dtype)
        self.model = torch.nn.Sequential(embedding, xformer)

    @staticmethod
    def from_config(config: NameSeqLearnerConfig, dtype: torch.dtype = torch.float32) -> 'NameSeqTransformer':
        return NameSeqTransformer(d_embedding=config.d_embedding,
                                  n_heads=config.n_heads,
                                  n_encoder_layers=config.n_encoder_layers,
                                  n_decoder_layers=config.n_decoder_layers,
                                  d_feedforward=config.d_feedforward,
                                  vocab_size=config.vocab_size,
                                  dtype=dtype)


class NameSeqLearnerApp(TrainableModelApp[NameSeqLearnerConfig, NameSeqTransformer]):
    def __init__(self, argv: Optional[list[str]] = None):
        super().__init__(NameSeqLearnerConfig(),
                         'Train a Transformer model to generate human names.',
                         argv=argv)
        self.tokenizer_dict: dict[str, int] = json.loads(self.config.tokenizer_dict_file.read_text())
        self.logger.info('Loaded token dictionary', n_tokens=len(self.tokenizer_dict))
        self.model: NameSeqTransformer | None = None

    def run(self):
        try:
            self.logger.info('Starting NameSeqLearner app with arguments', **asdict(self.config))
            self.model = NameSeqTransformer.from_config(self.config, dtype=self.dtype).to(self.device)
            self.model.eval()  # We're not training for the moment.
            data =
            data = TensorDataset(*generate_data(self.config.n_points, self.config.in_seq_length))
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
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
            # loss_fn = nn.CrossEntropyLoss()
            loss_fn = SequenceCrossEntropyLoss()
            train_loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True)
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