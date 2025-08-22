import pytest
from torch_playground.d03_basic_transformer import (
    HRBasicTransformer,
    BasicTransformerConfig,
    BasicTransformerApp,
    sieve,
    generate_data,
)
import torch


class TestD03BasicTransformer:

    def test_sieve_err(self):
        with pytest.raises(AssertionError):
            _ = sieve(-1)

    def test_sieve_edge_0(self):
        assert sieve(0) == [False]

    def test_sieve_edge_1(self):
        assert sieve(1) == [False, False]

    def test_sieve_ends_on_prime(self):
        assert sieve(23) == [False, False, True, True, False, True,
                                    False, True, False, False, False,
                                    True, False, True, False, False,
                                    False, True, False, True, False,
                                    False, False, True]

    def test_sieve_ends_on_composite(self):
        assert sieve(24) == [False, False, True, True, False, True,
                                    False, True, False, False, False,
                                    True, False, True, False, False,
                                    False, True, False, True, False,
                                    False, False, True, False]

    def test_sieve_square_n(self):
        assert sieve(16) == [False, False, True, True, False, True,
                                    False, True, False, False, False,
                                    True, False, True, False, False,
                                    False]

    def test_embedding_matrix_nearl_orthonormal(self):
        config = BasicTransformerConfig(d_model=8, vocab_size=5)
        model = HRBasicTransformer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
            d_feedforward=config.d_feedfoward,
            vocab_size=config.vocab_size
        )
        embedding_matrix = model.embedding_basis
        assert embedding_matrix.shape == (config.vocab_size, config.d_model)
        assert torch.allclose(embedding_matrix @ embedding_matrix.T,
                              torch.eye(config.vocab_size))