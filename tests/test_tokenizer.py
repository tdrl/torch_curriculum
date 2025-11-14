import pytest
from torch_playground.tokenizer import NGramTokenizer
import re


class TestTokenizer:

    @pytest.mark.parametrize('k', [1, 2, 10])
    def test_empty_k_gram(self, k: int):
        result = NGramTokenizer(k)
        assert result.n == k
        assert result.vocab_size() == 0
        assert result.tokens_to_ids == {}

    def test_force_add_unknown_token(self):
        result = NGramTokenizer(3)
        result.add_single_token('<UNKNOWN:___>', validate_token_length=False)
        assert result.vocab_size() == 1
        assert '<UNKNOWN:___>' in result.tokens_to_ids

    def test_validates_adding_long_token(self):
        result = NGramTokenizer(3)
        # This should pass.
        result.add_single_token('foo')
        assert result.vocab_size() == 1
        # This should fail.
        with pytest.raises(ValueError):
            result.add_single_token('uuddlrlr')

    @pytest.mark.parametrize(['k', 'text_len'], ((1, 1), (3, 3), (1, 3), (3, 1), (3, 5)))
    def test_non_empty_k_gram(self, k: int, text_len: int):
        raw_data = ''.join([chr(i + ord('a')) for i in range(text_len)])
        result = NGramTokenizer(k)
        print(f'Newly created tokenizer contents: {result.tokens_to_ids}')
        result.add_single_token(f'<UNKNOWN:{"_" * k}>', validate_token_length=False)
        n_added = result.add_to_token_dict(raw_data)
        expected_novel_tokens = max(text_len - k + 1, 1)
        assert n_added == expected_novel_tokens
        assert result.vocab_size() == expected_novel_tokens + 1, f'dict = {result.tokens_to_ids}'
        for g in result.tokens_to_ids:
            if result.tokens_to_ids[g] > 0:
                assert g in raw_data
            else:
                assert re.match(rf'<UNKNOWN:_{{{k}}}>', g)
        codes = set(result.tokens_to_ids.values())  # Dedup token values to check that all are unique.
        assert len(codes) == result.vocab_size()

    def test_non_unique_tokens(self):
        raw_data = 'abc' * 100
        result = NGramTokenizer(3)
        n_added = result.add_to_token_dict(raw_data)
        assert n_added == 3
        for expected_token in ('abc', 'bca', 'cab'):
            assert expected_token in result.tokens_to_ids
        assert result.vocab_size() == 3

    def test_specific_case_1(self):
        raw_data = 'abcd'
        result = NGramTokenizer(2)
        result.add_to_token_dict(raw_data)
        assert result.tokens_to_ids == {
            'ab': 0,
            'bc': 1,
            'cd': 2,
        }
