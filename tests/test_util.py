from dataclasses import dataclass, fields, field
from pathlib import Path
import pytest
import torch
import re
import json

from torch_playground.util import (
    parse_cmd_line_args,
    BaseConfiguration,
    setup_logging,
    accuracy,
    App,
    save_tensor,
    SequenceCrossEntropyLoss
)

class TestUtil:

    def test_parse_cmd_line_args_base(self):
        """Test parsing command line arguments with no arguments."""
        frozen_args = BaseConfiguration()  # Ensure we're not inspecting the same underlying object.
        args = parse_cmd_line_args(BaseConfiguration(), description=None, argv=[])
        assert isinstance(args, BaseConfiguration)
        expected_fields = fields(frozen_args)
        assert len(vars(args)) == len(expected_fields), f'Expected {len(expected_fields)} arguments to be parsed.'
        # Check default values for BaseArguments.
        for field in expected_fields:
            assert hasattr(args, field.name), f'Parsed arguments should have field {field}.'
            assert getattr(args, field.name) == field.default, f'Field {field.name} should have default value {field.default}.'

    def test_parse_cmd_line_args_base_can_override_shared_args(self, tmp_path):
        args = parse_cmd_line_args(BaseConfiguration(), description=None, argv=['--loglevel', 'DEBUG',
                                                                                '--output_dir', str(tmp_path),
                                                                                '--randseed=42',
                                                                                '--epochs=3',
                                                                                '--batch_size', '10'])
        assert isinstance(args, BaseConfiguration)
        assert args.loglevel == 'DEBUG', 'Log level should be overridden to DEBUG.'
        assert args.output_dir == tmp_path, f'Output directory should be overridden to {str(tmp_path)}.'
        assert args.randseed == 42
        assert args.epochs == 3
        assert args.batch_size == 10

    def test_parse_cmd_line_args_with_args_preserves_base_fields(self):
        """Test parsing command line arguments defaulted to provided vals."""
        @dataclass
        class CustomArgs(BaseConfiguration):
            arg1: str = 'default1'
            arg2: int = 42
        args = parse_cmd_line_args(CustomArgs(), description=None, argv=[])
        assert isinstance(args, CustomArgs)
        assert hasattr(args, 'arg1')
        assert hasattr(args, 'arg2')

    def test_parse_cmd_line_args_with_args_defaults(self):
        """Test parsing command line arguments defaulted to provided vals."""
        @dataclass
        class CustomArgs(BaseConfiguration):
            arg1: str = 'default1'
            arg2: int = 42
        args = parse_cmd_line_args(CustomArgs(), description=None, argv=[])
        assert isinstance(args, CustomArgs)
        assert hasattr(args, 'arg1')
        assert args.arg1 == 'default1'
        assert hasattr(args, 'arg2')
        assert args.arg2 == 42

    def test_parse_cmd_line_args_with_args_provided(self):
        """Test parsing command line arguments with provided values."""
        @dataclass
        class CustomArgs2(BaseConfiguration):
            arg21: str = 'default1'
            arg22: int = 42
        args = parse_cmd_line_args(CustomArgs2(), description='', argv=['--arg21', 'value1', '--arg22', '100'])
        assert isinstance(args, CustomArgs2)
        assert hasattr(args, 'arg21')
        assert args.arg21 == 'value1'
        assert hasattr(args, 'arg22')
        assert args.arg22 == 100

    def test_parse_cmd_line_args_prints_help(self, capsys):
        @dataclass
        class CustomArgs(BaseConfiguration):
            arg31: str = field(default='Twas brillig', metadata=BaseConfiguration._meta(help='A strange beeste'))
            arg32: int = field(default=7)  # No help provided
        with pytest.raises(SystemExit):
            _ = parse_cmd_line_args(CustomArgs(), description='Hunting of the Snark', argv=['--help'])
        stdout = capsys.readouterr().out
        assert 'Hunting of the Snark' in stdout
        assert re.search(r'--arg31.*A strange beeste\s+\(default: Twas brillig\)', stdout, re.IGNORECASE)
        assert re.search(r'--arg32.*7.*$', stdout, re.IGNORECASE)

    def test_logging_creates_files(self, tmp_path):
        """Test that logging creates files in the specified directory."""
        logdir = tmp_path / 'logs'
        logger = setup_logging(loglevel='DEBUG', logdir=logdir)
        logger.debug('This is a debug message')
        logger.info('This is an info message')
        assert logdir.exists(), f'Log directory {str(logdir)} should exist.'
        for logfile in ('debug.json.log', 'error.json.log'):
            assert (logdir / logfile).exists(), f'Log file {logfile} should exist in {str(logdir)}.'

    def test_logging_writes_to_files(self, tmp_path):
        logdir = tmp_path / 'logs'
        logger = setup_logging(loglevel='ERROR', logdir=logdir)
        logger.debug('Debug message to file')
        logger.info('Info message to file')
        logger.error('Error message to file')
        # Check that only the error message is in the error log file.
        err_log_contents = (logdir / 'error.json.log').read_text()
        assert 'Error message to file' in err_log_contents, 'Error message should be present in the error log file.'
        assert 'Debug message to file' not in err_log_contents, 'Debug message should not be present in the error log file.'
        assert 'Info message to file' not in err_log_contents, 'Info message should not be present in the error log file.'
        debug_log_contents = (logdir / 'debug.json.log').read_text()
        assert 'Error message to file' in debug_log_contents, 'Error message should be present in the debug log file.'
        assert 'Info message to file' in debug_log_contents, 'Info message should be present in the debug log file.'
        assert 'Debug message to file' in debug_log_contents, 'Debug message should be present in the debug log file.'

    def test_logging_writes_to_stderr(self, capsys, tmp_path):
        """Test that logging writes to stderr."""
        logdir = tmp_path / 'logs'
        logger = setup_logging(loglevel='DEBUG', logdir=logdir)
        logger.debug('Debug message to stderr')
        logger.info('Info message to stderr')
        captured = capsys.readouterr()
        assert 'Debug message to stderr' in captured.err
        assert 'Info message to stderr' in captured.err

    def test_accuracy(self):
        x = torch.as_tensor([1, 0, 3, 1, 2, 1, 3, 3, 1, 0])
        y = torch.as_tensor([9, 9, 9, 1, 2, 1, 3, 3, 1, 0])
        expected_acc = torch.as_tensor(0.7)
        assert torch.allclose(accuracy(x, y), expected_acc)

    def test_app_must_be_implemented(self):
        with pytest.raises(NotImplementedError):
            app = App(BaseConfiguration(), description='Test failing app', argv=[])
            app.run()

    def test_save_tensor_simple_tensor(self, tmp_path):
        out_data = torch.as_tensor([1., 2., 3.])
        dest: Path = tmp_path / 'out'
        save_tensor(out_data, dest)
        dest_pt = dest.with_suffix('.pt')
        dest_txt = dest.with_suffix('.txt')
        assert dest_pt.exists()
        assert dest_txt.exists()
        with dest_pt.open('rb') as d_in:
            in_data = torch.load(d_in)
            assert torch.allclose(out_data, in_data)
        in_text = dest_txt.read_text()
        assert re.match(r'\s*\[\s*1(.[0-9]*)?\s*,\s*2(.[0-9]*)?\s*,\s*3(.[0-9]*)?\s*\]', in_text)

    def test_app_init_saves_config(self, tmp_path):
        @dataclass
        class LocalConfig(BaseConfiguration):
            foo: int = field(default=7)
            bar: str = field(default='Twas brillig and the slithy toves')
        app = App(LocalConfig(), description='testing app', argv=['--output_dir', str(tmp_path),
                                                                  '--foo', '42',
                                                                  '--bar', 'uuddlrlr'])
        # Ensures that we're loading from file and not just reconsituting
        # defaults.
        expected = LocalConfig(foo=42, bar='uuddlrlr', output_dir=tmp_path)
        assert tmp_path in app.work_dir.parents
        config_file = app.work_dir / 'config.json'
        assert config_file.exists()
        actual = LocalConfig(**json.loads(config_file.read_text()))
        # This is a hack - not clear on the "right" way to auto-thunk
        # text to paths.
        actual.output_dir = Path(actual.output_dir)
        assert actual == expected

    @pytest.mark.parametrize(['batches', 'seq_len', 'n_categories'],
                             [(3, 10, 4), (1, 1, 2), (10, 1, 5), (10, 4, 2), (7, 5, 11)])
    def test_sequence_cross_entropy_handles_shapes(self, batches, seq_len, n_categories):
        loss_fn = SequenceCrossEntropyLoss()
        # Note on test data construction: We want things of the appropriate shape, but not all
        # zeros or ones. It's also nice to avoid randomness for test stability.
        # For the input_data, we'll just make every value unique and non-integral
        # by generating a numerical sequence and dividing it by a constant. For the target_data,
        # we need each entry to be in the range [0, n_categories) and we don't want the same entry
        # for each value of the input_data. So we again generate sequences in the range [0, n_categories)
        # and repeat them appropriately along sequence and batch dimension. The result is that
        # every pairing of input_data and target_data element should be unique.
        input_data = torch.arange(batches * seq_len * n_categories, dtype=torch.float32).reshape((batches, seq_len, n_categories)) / seq_len
        n_reps = seq_len // n_categories + 1
        target_data = torch.arange(n_categories).tile((batches, n_reps))[:, :seq_len]
        result = loss_fn(input_data, target_data)
        assert result.numel() == 1
        assert result.item() > 0