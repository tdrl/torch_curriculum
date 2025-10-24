import pytest
from torch_playground.build_tokenizer import TokenizerConfig, BuildTokenizerApp
from pathlib import Path
import json
import re


class TestBuildTokenizer:

    @staticmethod
    def setup_data(k: int, data: str | None, out_dir: Path) -> dict[str, int]:
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
        return json.loads(goal_file.read_text())

    @pytest.mark.parametrize('k', [1, 2, 10])
    def test_empty_k_gram(self, k: int, tmp_path: Path):
        result = self.setup_data(k=k, data=None, out_dir=tmp_path)
        assert result == {f'<UNKNOWN:{"_" * k}>': 0}

    @pytest.mark.parametrize(['k', 'text_len'], ((1, 1), (3, 3), (1, 3), (3, 1), (3, 5)))
    def test_non_empty_k_gram(self, k: int, text_len: int, tmp_path: Path):
        raw_data = ''.join([chr(i + ord('a')) for i in range(text_len)])
        print(f'Debug: text = "{raw_data}"')
        result = self.setup_data(k=k, data=raw_data, out_dir=tmp_path)
        assert len(result) == max(text_len - k + 1, 1) + 1
        for g in result:
            if result[g] > 0:
                assert g in raw_data
            else:
                assert re.match(rf'<UNKNOWN:_{{{k}}}>', g)
        codes = set(result.values())  # Dedup token values to check that all are unique.
        assert len(codes) == len(result)

    def test_specific_case_1(self, tmp_path: Path):
        raw_data = 'abcd'
        result = self.setup_data(k=2, data=raw_data, out_dir=tmp_path)
        assert result == {
            '<UNKNOWN:__>': 0,
            'ab': 1,
            'bc': 2,
            'cd': 3,
        }
