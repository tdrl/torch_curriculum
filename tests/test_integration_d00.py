import pytest
from torch_playground.d00_linear_projection import main, Arguments
from torch_playground.util import setup_logging


class TestIntegrarionD00Linear:

    def test_full_run(self, tmp_path):
        args = Arguments(input_dim=5, output_dim=4, n_samples=11, output_dir=tmp_path)
        logger = setup_logging(logdir=tmp_path / 'logs')
        main(logger=logger, args=args) # type: ignore
        expected_files = [
            'model.pth',
            'model.txt',
            'x.txt',
            'x.pt',
            'y.txt',
            'y.pt'
        ]
        out_dir = tmp_path / 'd00_linear_projection'
        for e in expected_files:
            assert (out_dir / e).exists()