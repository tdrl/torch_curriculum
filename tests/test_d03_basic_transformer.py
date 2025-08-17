import pytest
from torch_playground.d03_basic_transformer import (
    BasicTransformerConfig,
    BasicTransformerApp,
    sieve,
    generate_data,
)


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