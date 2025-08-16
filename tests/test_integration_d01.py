import pytest
from torch_playground.d01_linear_trainable import LinearTrainableApp


class TestIntegrationD01Linear:

    def test_full_run(self, tmp_path):
        app = LinearTrainableApp(['--dim', '10',
                                  '--num_train_samples', '9',
                                  '--epochs', '2',
                                  '--batch_size', '3',
                                  '--output_dir', str(tmp_path)])
        app.run()
        expected_files = [
            'config.json',
            'discriminator.txt',
            'discriminator.pt',
            'X.txt',
            'X.pt',
            'y.txt',
            'y.pt',
            'W_trained.txt',
            'W_trained.pt',
            'predictions_truth.txt',
            'predictions_truth.pt',
        ]
        out_dir = app.work_dir
        assert tmp_path in out_dir.parents
        for e in expected_files:
            assert (out_dir / e).exists()