"""Test suite for linear projection demo."""

import pytest
import torch
from torch.utils.data import DataLoader

from torch_playground.d00_linear_projection import HRLinear, create_data

class TestHRLinear:
    """Test cases for the HRLinear model."""

    @pytest.mark.parametrize('n_samples', [1, 5, 10, 1000])
    def test_data_creation_ds_size(self, n_samples):
        """Test if data creation returns a TensorDataset with correct shape."""
        input_dim = 11
        dataset = create_data(input_dim, n_samples)
        assert len(dataset) == n_samples
        assert dataset.tensors[0].shape == (n_samples, input_dim)

    @pytest.mark.parametrize('input_dim, output_dim', [(3, 3), (5, 2), (11, 7), (23, 33), (5, 1), (1, 1)])
    def test_model_initialization_random(self, input_dim, output_dim):
        """Test if the model initializes with random weights and biases."""
        model = HRLinear(input_dim, output_dim)
        assert model.W.shape == (input_dim, output_dim)
        assert model.b.shape == (output_dim,)
        assert model.W.dtype == torch.int32
        assert model.b.dtype == torch.int32

    def test_model_initialization_explicit(self):
        """Test if the model initializes with explicit weights and biases."""
        input_dim = 4
        output_dim = 3
        W = torch.as_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.int32)
        b = torch.as_tensor([1, 2, 3], dtype=torch.int32)
        model = HRLinear(input_dim, output_dim, W=W, b=b)
        assert torch.equal(model.W, W)
        assert torch.equal(model.b, b)

    @pytest.mark.parametrize('input_dim, output_dim', [(2, 2),  # b mismatch
                                                       (3, 2),  # W mismatch
                                                       (5, 4)]) # both mismatch
    def test_model_initialization_shape_mismatch(self, input_dim, output_dim):
        """Test if the model raises an assertion error for shape mismatch."""
        W = torch.as_tensor([[1, 2], [3, 4]], dtype=torch.int32)
        b = torch.as_tensor([1, 2, 3], dtype=torch.int32)  # Incorrect shape
        with pytest.raises(AssertionError):
            HRLinear(input_dim, output_dim, W=W, b=b)

    def test_forward_pass_shape(self):
        input_dim = 3
        output_dim = 2
        model = HRLinear(input_dim, output_dim)
        x = torch.randint(-10, 10, (5, input_dim), dtype=torch.int32)
        output = model(x)
        assert output.shape == (5, output_dim)

    def test_forward_pass_values(self):
        """Test the forward pass of the HRLinear model with known weights and biases."""
        input_dim = 2
        output_dim = 2
        W = torch.as_tensor([[1, 2], [3, 4]], dtype=torch.int32)
        b = torch.as_tensor([5, 6], dtype=torch.int32)
        model = HRLinear(input_dim, output_dim, W=W, b=b)
        x = torch.as_tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int32)
        output = model(x)
        expected_output = torch.as_tensor([[7*1 + 8*3 + 5, 7*2 + 8*4 + 6],
                                           [9*1 + 10*3 + 5, 9*2 + 10*4 + 6],
                                           [11*1 + 12*3 + 5, 11*2 + 12*4 + 6]], dtype=torch.int32)
        assert torch.equal(output, expected_output), f'Expected {expected_output}, got {output}'


    def test_iterating_over_dataset(self):
        """Test if iterating over the dataset returns correct data."""
        input_dim = 3
        n_samples = 100
        batch_size = 5
        dataset = create_data(input_dim, n_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for idx, batch in enumerate(dataloader):
            assert batch[0].shape[1] == input_dim, f'Batch shape mismatch: {batch[0].shape}'
            assert batch[0].dtype == torch.int32, f'Batch dtype mismatch: {batch[0].dtype}'
            assert len(batch[0]) == batch_size, f'Expected batch size {batch_size}, got {len(batch[0])} (batch {idx})'