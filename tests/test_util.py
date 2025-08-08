from dataclasses import dataclass, fields
from pathlib import Path
import pytest

from torch_playground.util import parse_cmd_line_args, BaseArguments, setup_logging

class TestUtil:

    def test_parse_cmd_line_args_base(self):
        """Test parsing command line arguments with no arguments."""
        frozen_args = BaseArguments()  # Ensure we're not inspecting the same underlying object.
        args = parse_cmd_line_args(BaseArguments(), description=None, argv=[])
        assert isinstance(args, BaseArguments)
        expected_fields = fields(frozen_args)
        assert len(vars(args)) == len(expected_fields), f'Expected {len(expected_fields)} arguments to be parsed.'
        # Check default values for BaseArguments.
        for field in expected_fields:
            assert hasattr(args, field.name), f'Parsed arguments should have field {field}.'
            assert getattr(args, field.name) == field.default, f'Field {field.name} should have default value {field.default}.'

    def test_parse_cmd_line_args_base_can_override_shared_args(self):
        args = parse_cmd_line_args(BaseArguments(), description=None, argv=['--loglevel', 'DEBUG', '--logdir', '/tmp/logs'])
        assert isinstance(args, BaseArguments)
        assert args.loglevel == 'DEBUG', 'Log level should be overridden to DEBUG.'
        assert args.logdir == Path('/tmp/logs'), 'Log directory should be overridden to /tmp/logs.'

    def test_parse_cmd_line_args_with_args_preserves_base_fields(self):
        """Test parsing command line arguments defaulted to provided vals."""
        @dataclass
        class CustomArgs(BaseArguments):
            arg1: str = 'default1'
            arg2: int = 42
        args = parse_cmd_line_args(CustomArgs(), description=None, argv=[])
        assert isinstance(args, CustomArgs)
        assert hasattr(args, 'loglevel')
        assert hasattr(args, 'logdir')

    def test_parse_cmd_line_args_with_args_defaults(self):
        """Test parsing command line arguments defaulted to provided vals."""
        @dataclass
        class CustomArgs(BaseArguments):
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
        class CustomArgs2(BaseArguments):
            arg21: str = 'default1'
            arg22: int = 42
        args = parse_cmd_line_args(CustomArgs2(), description='', argv=['--arg21', 'value1', '--arg22', '100'])
        assert isinstance(args, CustomArgs2)
        assert hasattr(args, 'arg21')
        assert args.arg21 == 'value1'
        assert hasattr(args, 'arg22')
        assert args.arg22 == 100

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