"""Parameterize a tokenizer from a fixed data file."""

from torch_playground.util import BaseConfiguration, BaseApp, FileDataset
from torch_playground.tokenizer import NGramTokenizer
from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import DataLoader

@dataclass
class TokenizerConfig(BaseConfiguration):
    data_file: Path = field(default=Path('/dev/null'),
                            metadata=BaseConfiguration._meta(help='File of newline-separated strings to build tokenizer from.',
                                                             required=True))
    ngram_len: int = field(default=1,
                           metadata=BaseConfiguration._meta(help='Length of ngrams, in characters, to tokenize.'))


class BuildTokenizerApp(BaseApp[TokenizerConfig]):
    """An application that creates a tokenizer and saves its model.

    This reads data from the file given in the config.data_file and writes the resulting token
    mapping model to config.output_dir / token_dict.n={self.config.ngram_len}.json."""

    def run(self):
        data = DataLoader(dataset=FileDataset(self.config.data_file).with_transform(lambda x: x.strip()),
                          batch_size=self.config.batch_size)
        tokenizer = NGramTokenizer(self.config.ngram_len)
        unknown_token = f'<UNKNOWN:{"_" * self.config.ngram_len}>'  # Guaranteed never to be an n-gram.
        tokenizer.add_single_token(unknown_token, validate_token_length=False)
        for batch in data:
            for row in batch:
                tokenizer.add_to_token_dict(row)
        self.logger.info('Finished tokenizing', n_tokens=tokenizer.vocab_size())
        tokenizer_file = (self.config.output_dir / f'token_dict.n={self.config.ngram_len}.json')
        tokenizer.to_file(tokenizer_file)
        self.logger.info('Wrote tokenizer state file', tokenizer_file=str(tokenizer_file))


def main(argv: list[str] | None = None):
    BuildTokenizerApp(arg_template=TokenizerConfig(),
                      description='Build a tokenizer dictionary from a file of lines',
                      argv=argv).run()


if __name__ == '__main__':
    main()
