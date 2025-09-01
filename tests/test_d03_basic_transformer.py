import pytest
from torch_playground.d03_basic_transformer import (
    HRLBasicTransformer,
    BasicTransformerConfig,
    BasicTransformerApp,
    sieve,
    generate_data,
)
import torch
from dataclasses import asdict
from typing import Any


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

    def test_generate_data(self):
        x, y = generate_data(7, 4)
        assert torch.allclose(x, torch.as_tensor([[0, 1, 2, 3],
                                                  [1, 2, 3, 4],
                                                  [2, 3, 4, 5],
                                                  [3, 4, 5, 6],
                                                  [4, 5, 6, 7],
                                                  [5, 6, 7, 8],
                                                  [6, 7, 8, 9]], dtype=x.dtype))
        assert torch.allclose(y, torch.as_tensor([[0, 0, 1, 1],
                                                  [0, 1, 1, 0],
                                                  [1, 1, 0, 1],
                                                  [1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [1, 0, 1, 0],
                                                  [0, 1, 0, 0]], dtype=y.dtype))


    def test_embedding_matrix_nearly_orthonormal(self):
        config = BasicTransformerConfig(d_model=128, vocab_size=512)
        model = HRLBasicTransformer.from_config(config=config)
        embedding_matrix = model.embedding_mapping
        assert embedding_matrix.shape == (config.vocab_size, config.d_model)
        es = embedding_matrix @ embedding_matrix.T
        assert torch.allclose(torch.diag(es), torch.ones((config.vocab_size,)))
        esp = torch.abs(es - torch.eye(config.vocab_size))
        assert torch.max(esp) < 0.5  # Note: only holds stochastically, but with Pr -> 1 asymptotically with d_model.

    def test_embedding_mechanics(self):
        # We're not trying to prove near-orthogonality here, so we can get
        # away with a smaller d_model for testing purposes. Choose
        # vocab size to be relatively prime to d_model just so we're sure
        # we're not accidentally conflating the two.
        d_model = 64
        vocab_size = 271
        config = BasicTransformerConfig(d_model=d_model, vocab_size=vocab_size)
        model = HRLBasicTransformer.from_config(config)
        batch_size = 7
        seq_length = 149
        data = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_length))
        actual = model.embed(data)
        assert actual.shape == (batch_size, seq_length, d_model)
        for b in range(batch_size):
            for i in range(10):
                assert torch.allclose(actual[b, i, :], model.embedding_mapping[data[b, i], :])

    def test_decoding_shapes(self):
        d_model = 64
        vocab_size = 271
        config = BasicTransformerConfig(d_model=d_model, vocab_size=vocab_size)
        model = HRLBasicTransformer.from_config(config)
        batch_size = 7
        seq_length = 149
        data = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_length))
        embedded = model.embed(data)
        actual = model.decode(embedded)
        assert actual.shape == (batch_size, seq_length, vocab_size)

    def test_decoding_semantics(self):
        d_model = 9
        # Choose vocab_size (a) < d_model, so we can have a complete basis, and
        # (b) relatively prime to d_model, so we're sure we're not confusing dimensions.
        vocab_size = 5
        batch_size = 7
        seq_len = 3
        config = BasicTransformerConfig(d_model=d_model, vocab_size=vocab_size, in_seq_length=seq_len, out_seq_length=seq_len, n_heads=3)
        model = HRLBasicTransformer.from_config(config=config)
        # Override initializer for embedding so that we can predict the cosine similarity.
        model.embedding_mapping = torch.zeros(size=(vocab_size, d_model), dtype=torch.float32)
        for v in range(vocab_size):
            model.embedding_mapping[v, v] = 1.0
        src_data_embedded = torch.zeros(size=(batch_size, seq_len, d_model))
        for b in range(batch_size):
            for s in range(seq_len):
                src_data_embedded[b, s] = (b + 1) * (s + 1) * torch.arange(d_model, dtype=torch.float32)
        actual = model.decode(src_data_embedded)
        assert actual.shape == (batch_size, seq_len, vocab_size)
        for b in range(batch_size):
            for s in range(seq_len):
                for v in range(vocab_size):
                    # Note 1: embedding_mapping[v, :] dot src_data_embedded[b, s] == src_data_embedded[b, s, v]
                    #                                                             == (b + 1) * (s + 1) * v,
                    # by construction b/c embedding_mapping[v, :] is a unit basis vector along dim v.
                    # Note 2: || embedding_mapping[v, :] || == 1.0 by construction, for the same reason.
                    # Hence: Full cosine sim devolves to this special form in this case.
                    assert torch.allclose(actual[b, s, v], (b + 1) * (s + 1) * v / torch.linalg.vector_norm(src_data_embedded[b, s]))

    def test_transformer_in_out(self):
        d_model = 9
        vocab_size = 5
        batch_size = 7
        in_seq_len = 11
        out_seq_len = 8
        config = BasicTransformerConfig(d_model=d_model,
                                        vocab_size=vocab_size,
                                        in_seq_length=in_seq_len,
                                        out_seq_length=out_seq_len,
                                        n_heads=3)
        model = HRLBasicTransformer.from_config(config=config)
        src_data = torch.randn(size=(batch_size, in_seq_len, d_model))
        tgt_data = torch.randn(size=(batch_size, out_seq_len, d_model))
        result: torch.Tensor = model.xformer(src_data, tgt_data)
        assert result.shape == tgt_data.shape

    @pytest.mark.parametrize(['d_model', 'vocab_size', 'batch_size', 'in_seq_length', 'out_seq_length'],
                             ((32, 271, 7, 149, 93),
                              (64, 101, 13, 217, 181),
                              (128, 97, 7, 149, 57),
                              (72, 91, 13, 217, 101)))
    def test_end_to_end_shape(self, d_model, vocab_size, batch_size, in_seq_length, out_seq_length):
        config = BasicTransformerConfig(d_model=d_model,
                                        vocab_size=vocab_size,
                                        n_heads=8,
                                        in_seq_length=in_seq_length,
                                        out_seq_length=out_seq_length)
        model = HRLBasicTransformer.from_config(config)
        src = torch.randint(low=0, high=vocab_size, size=(batch_size, in_seq_length))
        target = torch.randint(low=0, high=vocab_size, size=(batch_size, out_seq_length))
        actual = model(src, target)
        assert actual.shape == (batch_size, out_seq_length, vocab_size)
