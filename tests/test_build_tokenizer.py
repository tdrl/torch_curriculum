import pytest
from torch_playground.build_tokenizer import TokenizerConfig, BuildTokenizerApp
from torch_playground.tokenizer import NGramTokenizer
from pathlib import Path
import json
import re


class TestBuildTokenizer:

    @staticmethod
    def setup_data(k: int, data: str | None, out_dir: Path) -> NGramTokenizer:
        dest_file = out_dir / f'data_{k}.txt'
        if data is None:
            dest_file.write_bytes(bytes())
        else:
            dest_file.write_text(data)
        args = ['--data_file', str(dest_file),
                '--ngram_len', str(k),
                '--output_dir', str(out_dir)]
        BuildTokenizerApp(TokenizerConfig(), argv=args, description='test run').run()
        goal_file = out_dir / f'token_dict.n={k}.json'
        assert goal_file.exists()
        return NGramTokenizer.from_file(goal_file)

    @pytest.mark.parametrize('k', [1, 2, 10])
    def test_empty_k_gram(self, k: int, tmp_path: Path):
        result = self.setup_data(k=k, data=None, out_dir=tmp_path)
        assert result.vocab_size() == 2
        assert result.unknown_token == f'<UNKNOWN:{"_" * k}>'
        assert result.padding_token == '<PAD>'
        assert result.tokens_to_ids == {
            f'<UNKNOWN:{"_" * k}>': 0,
            '<PAD>': 1,
        }

    @pytest.mark.parametrize(['k', 'text_len'], ((1, 1), (3, 3), (1, 3), (3, 1), (3, 5)))
    def test_non_empty_k_gram(self, k: int, text_len: int, tmp_path: Path):
        raw_data = ''.join([chr(i + ord('a')) for i in range(text_len)])
        print(f'Debug: text = "{raw_data}"')
        result = self.setup_data(k=k, data=raw_data, out_dir=tmp_path)
        assert result.vocab_size() == max(text_len - k + 1, 1) + 2  # n-grams + unknown + padding
        for t, v in result.tokens_to_ids.items():
            if v == 0:
                assert re.match(rf'<UNKNOWN:_{{{k}}}>', t)
            elif v == 1:
                assert t == '<PAD>'
            else:  # Not unknown or padding
                assert t in raw_data
        codes = set(result.tokens_to_ids.values())  # Dedup token values to check that all are unique.
        assert len(codes) == result.vocab_size()

    def test_specific_case_1(self, tmp_path: Path):
        raw_data = 'abcd'
        result = self.setup_data(k=2, data=raw_data, out_dir=tmp_path)
        assert result.tokens_to_ids == {
            '<UNKNOWN:__>': 0,
            '<PAD>': 1,
            'ab': 2,
            'bc': 3,
            'cd': 4,
        }
