"""Test suite for linear projection demo."""

from torch_playground.d00_linear_projection import HRLinear, create_data

class TestHRLinear:
    """Test cases for the HRLinear model."""

    def test_data_creation(self):
        """Test if data creation returns a TensorDataset with correct shape."""
        input_dim = 3
        n_samples = 10
        dataset = create_data(input_dim, n_samples)
        assert len(dataset) == n_samples
        assert dataset.tensors[0].shape == (n_samples, input_dim)

    def test_forward_pass(self):
        """Test the forward pass of the HRLinear model."""
        input_dim = 3
        output_dim = 2
        model = HRLinear(input_dim, output_dim)
        x = torch.randint(-10, 10, (5, input_dim), dtype=torch.int32)
        output = model(x)
        assert output.shape == (5, output_dim)

    def test_model_initialization(self):
        """Test if the model initializes with random weights and biases."""
        input_dim = 3
        output_dim = 2
        model = HRLinear(input_dim, output_dim)
        assert model.W.shape == (input_dim, output_dim)
        assert model.b.shape == (output_dim,)
        assert model.W.dtype == torch.int32
        assert model.b.dtype == torch.int32