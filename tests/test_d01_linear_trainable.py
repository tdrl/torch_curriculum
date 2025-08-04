"""Tests for d01_linear_trainable.py"""

from torch_playground.d01_linear_trainable import LinearTrainableApp
import pytest

class TestLinearTrainableApp:

    @pytest.mark.parametrize('dim', [1, 5, 10])
    @pytest.mark.parametrize('num_samples', [10, 50, 100])
    def test_generate_data(self, dim: int, num_samples: int):
        """Test that data generation works and produces the expected shapes."""
        app = LinearTrainableApp(argv=[])
        app.args.dim = dim
        app.args.num_train_samples = num_samples
        data, discriminator = app.create_data()
        X, y = data.tensors
        assert X.shape == (num_samples, dim), f'Expected X shape ({num_samples}, {dim}), got {X.shape}'
        assert y.shape == (num_samples,), f'Expected y shape ({num_samples},), got {y.shape}'
        assert discriminator.shape == (dim,), f'Expected discriminator shape ({dim},), got {discriminator.shape}'
        # Check that y contains only -1 and 1
        assert set(y.tolist()).issubset({-1, 1}), f'Expected y to contain only -1 and 1, got {set(y.tolist())}'