import pytest
import torch
from torch_playground.d02_linear_multilayer import DataGenerator

class TestLinearMultilayer:

    @pytest.mark.parametrize('dim', [1, 5, 50])
    @pytest.mark.parametrize('n_classes', [2, 5, 20])
    def test_data_generator_init_hyperparameter_shapes(self, dim, n_classes):
        """Check that the generator produces the expected shapes in its hyperparameters."""
        # Positive Definiteness test: The Cholesky factorization in DataGenerator.init that computes
        # the cached covariance matrix factors throws RuntimeError if its matrix arg is not PD.
        try:
            dg = DataGenerator(dim=dim, n_classes=n_classes, dtype=torch.float32)
        except RuntimeError as e:
            raise AssertionError(f'Class: Expected covariances to be PD, but failed chol() during initialization', e)
        assert dg.means.shape == (n_classes, dim)
        assert dg.covariances.shape == (n_classes, dim, dim)
        assert dg._cov_factors_cache.shape == (n_classes, dim, dim)
        assert dg.means.shape == (n_classes, dim), f'Expected mean dimension = {(n_classes, dim)}; was dimension = {dg.means.shape}'

    @pytest.mark.parametrize('dim', [3, 7])
    @pytest.mark.parametrize('n_points', [1, 10, 100])
    @pytest.mark.parametrize('n_classes', [1, 5, 10])
    def test_data_generator_produces_correct_shapes(self, dim, n_points, n_classes):
        dg = DataGenerator(dim=dim, n_classes=n_classes, dtype=torch.float32)
        data = dg.generate(n_points=n_points)
        assert data.tensors[0].shape == (n_points, dim)
        assert data.tensors[1].shape == (n_points,)
