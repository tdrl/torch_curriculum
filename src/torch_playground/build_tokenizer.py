"""Parameterize a tokenizer from a fixed data file."""

from typing import Iterator
from torch_playground.util import BaseConfiguration, BaseApp
from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader, IterableDataset
import json

@dataclass
class TokenizerConfig(BaseConfiguration):
    data_file: Path = field(default=Path('/dev/null'),
                            metadata=BaseConfiguration._meta(help='File of newline-separated strings to build tokenizer from.',
                                                             required=True))
    ngram_len: int = field(default=1,
                           metadata=BaseConfiguration._meta(help='Length of ngrams, in characters, to tokenize.'))


class UniqueIdFactory:
    def __init__(self, init_id: int = 0) -> None:
        self.id = init_id

    def __call__(self) -> int:
        result = self.id
        self.id += 1
        return result


class FileDataset(IterableDataset):
    def __init__(self, data_file: Path) -> None:
        super().__init__()
        self.data_file = data_file
        self.data_handle = data_file.open('rt', encoding='utf-8', buffering=(1 << 20))

    def __iter__(self) -> Iterator:
        return self.data_handle


class BuildTokenizerApp(BaseApp[TokenizerConfig]):
    """An application that creates a tokenizer and saves its dict."""

    def run(self):
        # with self.config.data_file.open('rt', encoding='utf-8', buffering=(1 << 20)) as data_handle:
        data = DataLoader(dataset=FileDataset(self.config.data_file), batch_size=self.config.batch_size)
        tokenizer = dict()
        id_factory = UniqueIdFactory()
        for batch in data:
            for row in batch:
                row = row.strip()
                for i in range(len(row) - self.config.ngram_len + 1):
                    gram = row[i:(i + self.config.ngram_len)]
                    if gram not in tokenizer:
                        tokenizer[gram] = id_factory()
        self.logger.info('Finished tokenizing', n_tokens=len(tokenizer))
        dict_file = (self.config.output_dir / f'token_dict.n={self.config.ngram_len}.json')
        dict_file.write_text(json.dumps(tokenizer, indent=2))
        self.logger.info('Token dict location', dict_file=str(dict_file))


def main(argv: list[str] | None = None):
    BuildTokenizerApp(arg_template=TokenizerConfig(),
                      description='Build a tokenizer dictionary from a file of lines',
                      argv=argv).run()


if __name__ == '__main__':
    main()
