import pytest
from torch_playground.d02_linear_multilayer import LinearTrainableApp


class TestIntegrationD02Linear:

    def test_full_run(self, tmp_path):
        app = LinearTrainableApp(['--dim', '10',
                                  '--n_classes', '3',
                                  '--n_hidden_layers', '2',
                                  '--n_train_samples', '9',
                                  '--n_val_samples', '10',
                                  '--epochs', '2',
                                  '--batch_size', '3',
                                  '--output_dir', str(tmp_path)])
        app.run()
        expected_files = [
            'config.json',
            'hyper_cov.txt',
            'hyper_cov.pt',
            'hyper_means.txt',
            'hyper_means.pt',
            'X.txt',
            'X.pt',
            'y.txt',
            'y.pt',
            'trained_model.pt',
            'predictions_truth.txt',
            'predictions_truth.pt',
            'val_predictions_truth.txt',
            'val_predictions_truth.pt',
        ]
        out_dir = app.work_dir
        assert tmp_path in out_dir.parents
        for e in expected_files:
            assert (out_dir / e).exists()