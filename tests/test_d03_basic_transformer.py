import pytest
from torch_playground.d03_basic_transformer import (
    HRLBasicTransformer,
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

    def test_embedding_matrix_nearly_orthonormal(self):
        config = BasicTransformerConfig(d_model=128, vocab_size=512)
        model = HRLBasicTransformer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
            d_feedforward=config.d_feedfoward,
            vocab_size=config.vocab_size
        )
        embedding_matrix = model.embedding_mapping
        assert embedding_matrix.shape == (config.vocab_size, config.d_model)
        es = embedding_matrix @ embedding_matrix.T
        assert torch.allclose(torch.diag(es), torch.ones((config.vocab_size,)))
        esp = torch.abs(es - torch.eye(config.vocab_size))
        assert torch.max(esp) < 0.5  # Note: only holds stochastically, but with Pr -> 1 asymptotically with d_model.
