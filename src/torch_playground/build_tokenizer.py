"""Parameterize a tokenizer from a fixed data file."""

from torch_playground.util import BaseConfiguration, TrainableModelApp
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TokenizerConfig(BaseConfiguration):
    data_file: Path | None = field(default=None,
                                   metadata=BaseConfiguration._meta(help='File of newline-separated strings to build tokenizer from.',
                                                                    required=True))
