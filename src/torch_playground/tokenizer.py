"""Simple string tokenizer class."""

from pathlib import Path
import json


class _UniqueIdFactory:
    """Facory object to generate unique, sequential IDs.

    Every time you invoke an instance of this class (via call), you'll get back a unique ID.

    Arguments:
        init_id (int): Starting value for ID sequence.
    """
    def __init__(self, init_id: int = 0) -> None:
        self.id = init_id

    def __call__(self) -> int:
        result = self.id
        self.id += 1
        return result


# Open question (HRL): Should this be a subclass of nn.Model? Would that allow us to cleanly
# chain it into the model? Let's implement it in the basic way first and then try retrofitting it.
class NGramTokenizer(object):
    def __init__(self, n: int,
                 initial_token_map: dict[str, int] | None = None,
                 unknown_token: str | None = None,
                 padding_token: str | None = None) -> None:
        self.n = n
        self.tokens_to_ids = initial_token_map if initial_token_map is not None else {}
        if len(self.tokens_to_ids) == 0:
            self.id_factory = _UniqueIdFactory()
        else:
            self.id_factory = _UniqueIdFactory(init_id=max(self.tokens_to_ids.values()) + 1)
        self.unknown_token = unknown_token
        self.padding_token = padding_token

    def to_file(self, file: Path):
        file.write_text(json.dumps({
            'n': self.n,
            'token_dict': self.tokens_to_ids,
            'unknown_token': self.unknown_token,
            'padding_token': self.padding_token,
        }))

    @staticmethod
    def from_file(file: Path) -> 'NGramTokenizer':
        data = json.loads(file.read_text())
        assert 'n' in data and isinstance(data['n'], int)
        assert 'token_dict' in data and isinstance(data['token_dict'], dict)
        return NGramTokenizer(n=data['n'],
                              initial_token_map=data['token_dict'],
                              unknown_token=data.get('unknown_token', None),
                              padding_token=data.get('padding_token', None))

    def _ngrams(self, data: str):
        for i in range(max(len(data) - self.n + 1, 1)):
            yield data[i:(i + self.n)]

    def vocab_size(self) -> int:
        """Return the number of distinct tokens in the vocabulary."""
        return len(self.tokens_to_ids)

    def last_used_id(self) -> int:
        """Return the last used (i.e., maximum) token value in the current vocabulary."""
        return max(self.tokens_to_ids.values())

    def tokenize(self, data: str) -> list[int]:
        """Tokenize a single string into a list of NGram token IDs.

        Args:
            data (str): Data to split into tokens.
        """
        return [self.tokens_to_ids[g] for g in self._ngrams(data)]

    def add_single_token(self, token: str, validate_token_length: bool = True):
        if validate_token_length and len(token) > self.n:
            raise ValueError(f'Tried to add a too-large token "{token}" (len == {len(token)}) '
                             f'to NGramTokenizer with n={self.n}. Token too big to swallow! Sorry!')
        self.tokens_to_ids[token] = self.id_factory()

    def add_unknown_token(self, token: str):
        """Designate a special unknown token to be used when tokenizing unknown n-grams.

        Args:
            token (str): Token string to use as the unknown token.
        """
        self.add_single_token(token, validate_token_length=False)
        self.unknown_token = token

    def add_padding_token(self, token: str):
        """Designate a special padding token to be used when tokenizing sequences.

        Args:
            token (str): Token string to use as the padding token.
        """
        self.add_single_token(token, validate_token_length=False)
        self.padding_token = token

    def add_to_token_dict(self, data: str) -> int:
        """Split data into NGram tokens and add any new tokens to the internal token-to-id dictionary.

        Args:
            data (str): Data to parse for novel tokens.

        Returns:
            int: Number of novel tokens added to the dictionary.
        """
        tokens_added = 0
        for g in self._ngrams(data):
            if g not in self.tokens_to_ids:
                self.add_single_token(g)
                tokens_added += 1
        return tokens_added

    def get_token_id(self, token: str) -> int | None:
        """Get the ID for a given token string.

        Args:
            token (str): Token string to look up.
        Returns:
            int | None: ID of the token, or None if the token is not in the dictionary.
        """
        return self.tokens_to_ids.get(token, None)
