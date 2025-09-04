import pytest
from pathlib import Path
from torch_playground.d03_basic_transformer import BasicTransformerApp


@pytest.mark.integration
class TestIntegrationD03BasicTransformer:

    def test_full_run(self, tmp_path):
        app = BasicTransformerApp(['--output_dir', str(tmp_path),
                                   '--epochs', '2',
                                   '--batch_size', '3',
                                   '--d_model', '32',
                                   '--n_heads', '4',
                                   '--n_encoder_layers', '2',
                                   '--n_decoder_layers', '2',
                                   '--d_feedforward', '16',
                                   '--in_seq_length', '50',
                                   '--out_seq_length', '50',
                                   '--vocab_size', '60',  # At least in_seq_length + n_points
                                   '--n_points', '9'])
        app.run()
        expected_files = [
            'config.json',
            'in_seq_data.txt',
            'in_seq_data.pt',
            'out_seq_data.txt',
            'out_seq_data.pt',
            'test_indices.txt',
            'train_indices.txt',
            'val_indices.txt',
            'model_summary.txt',
            'trained_model.pt',
        ]
        out_dir = app.work_dir
        assert tmp_path in out_dir.parents
        for e in expected_files:
            assert (out_dir / e).exists()