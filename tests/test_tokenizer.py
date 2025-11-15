import pytest
from torch_playground.tokenizer import NGramTokenizer
import re
import torch


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

    # ==================== Decoding Tests ====================

    def test_decode_single_token(self):
        """Test decoding a single token ID."""
        tokenizer = NGramTokenizer(2)
        tokenizer.add_single_token('ab')
        result = tokenizer.decode([0])
        assert result == 'ab'

    def test_decode_sequence(self):
        """Test decoding multiple token IDs."""
        tokenizer = NGramTokenizer(2)
        tokenizer.add_to_token_dict('abcd')
        # Should have tokens: 'ab'->0, 'bc'->1, 'cd'->2
        result = tokenizer.decode([0, 1, 2])
        assert result == 'abbccd'

    def test_decode_tensor_input(self):
        """Test decoding from torch.Tensor input."""
        tokenizer = NGramTokenizer(2)
        tokenizer.add_to_token_dict('abc')
        # Should have tokens: 'ab'->0, 'bc'->1
        token_tensor = torch.tensor([0, 1], dtype=torch.long)
        result = tokenizer.decode(token_tensor)
        assert result == 'abbc'

    def test_decode_tensor_1d(self):
        """Test decoding from 1D torch.Tensor."""
        tokenizer = NGramTokenizer(1)
        tokenizer.add_to_token_dict('abc')
        # Should have tokens: 'a'->0, 'b'->1, 'c'->2
        token_tensor = torch.tensor([0, 1, 2], dtype=torch.long)
        result = tokenizer.decode(token_tensor)
        assert result == 'abc'

    def test_decode_invalid_token_id_raises_error(self):
        """Test that decoding with invalid token ID raises ValueError."""
        tokenizer = NGramTokenizer(2)
        tokenizer.add_single_token('ab')
        with pytest.raises(ValueError, match='Token ID .* not found in vocabulary'):
            tokenizer.decode([0, 999])  # 999 doesn't exist

    def test_decode_empty_list(self):
        """Test decoding an empty token list."""
        tokenizer = NGramTokenizer(2)
        tokenizer.add_single_token('ab')
        result = tokenizer.decode([])
        assert result == ''

    def test_single_char_tokenizer_round_trip(self):
        """Test round-trip with single-character tokens (true preservation)."""
        raw_data = 'hello'
        tokenizer = NGramTokenizer(1)  # Single character tokens
        tokenizer.add_to_token_dict(raw_data)
        # Encode: 'hello' -> [id(h), id(e), id(l), id(l), id(o)]
        token_ids = tokenizer.tokenize(raw_data)
        # Decode: should recover original
        decoded = tokenizer.decode(token_ids)
        assert decoded == raw_data

    def test_round_trip_encode_decode(self):
        """Test that encode → decode produces valid output for multi-char tokens."""
        raw_data = 'hello'
        tokenizer = NGramTokenizer(2)
        tokenizer.add_to_token_dict(raw_data)
        # Encode: 'hello' -> ids of ['he', 'el', 'll', 'lo']
        token_ids = tokenizer.tokenize(raw_data)
        # Decode: concatenate tokens -> 'heellllo'
        decoded = tokenizer.decode(token_ids)
        # Output is valid but different due to n-gram overlap
        assert len(decoded) > 0
        assert isinstance(decoded, str)
        # Verify consistency: same tokens should always decode the same way
        decoded_again = tokenizer.decode(token_ids)
        assert decoded == decoded_again

    def test_round_trip_with_longer_text(self):
        """Test single-char round-trip with longer text."""
        raw_data = 'abracadabra'
        tokenizer = NGramTokenizer(1)  # Single char tokens preserve original
        tokenizer.add_to_token_dict(raw_data)
        token_ids = tokenizer.tokenize(raw_data)
        decoded = tokenizer.decode(token_ids)
        assert decoded == raw_data

    # ==================== Token Lookup Tests ====================

    def test_get_token_string_valid(self):
        """Test looking up string for valid token ID."""
        tokenizer = NGramTokenizer(2)
        tokenizer.add_single_token('ab')
        result = tokenizer.get_token_string(0)
        assert result == 'ab'

    def test_get_token_string_invalid_returns_none(self):
        """Test that invalid token ID returns None."""
        tokenizer = NGramTokenizer(2)
        tokenizer.add_single_token('ab')
        result = tokenizer.get_token_string(999)
        assert result is None

    def test_get_token_string_multiple_tokens(self):
        """Test token lookup with multiple tokens in vocab."""
        tokenizer = NGramTokenizer(2)
        tokenizer.add_to_token_dict('abcd')
        # Should have 'ab'->0, 'bc'->1, 'cd'->2
        assert tokenizer.get_token_string(0) == 'ab'
        assert tokenizer.get_token_string(1) == 'bc'
        assert tokenizer.get_token_string(2) == 'cd'

    def test_has_token_id_valid(self):
        """Test has_token_id returns True for valid IDs."""
        tokenizer = NGramTokenizer(2)
        tokenizer.add_to_token_dict('abcd')
        assert tokenizer.has_token_id(0) is True
        assert tokenizer.has_token_id(1) is True
        assert tokenizer.has_token_id(2) is True

    def test_has_token_id_invalid(self):
        """Test has_token_id returns False for invalid IDs."""
        tokenizer = NGramTokenizer(2)
        tokenizer.add_to_token_dict('abcd')
        assert tokenizer.has_token_id(999) is False
        assert tokenizer.has_token_id(-1) is False
        assert tokenizer.has_token_id(3) is False

    def test_has_token_id_empty_vocab(self):
        """Test has_token_id with empty vocabulary."""
        tokenizer = NGramTokenizer(2)
        assert tokenizer.has_token_id(0) is False
        assert tokenizer.has_token_id(1) is False

    def test_decode_with_unknown_and_padding_tokens(self):
        """Test decoding with special tokens (unknown, padding)."""
        tokenizer = NGramTokenizer(3)
        tokenizer.add_unknown_token('<UNK>')
        tokenizer.add_padding_token('<PAD>')
        tokenizer.add_to_token_dict('hello')
        # Get the IDs
        unk_id = tokenizer.get_token_id('<UNK>')
        pad_id = tokenizer.get_token_id('<PAD>')
        # Decode a mix
        token_ids = [unk_id, pad_id, 0]  # <UNK>, <PAD>, first token of 'hello'
        result = tokenizer.decode(token_ids)
        assert '<UNK>' in result
        assert '<PAD>' in result
