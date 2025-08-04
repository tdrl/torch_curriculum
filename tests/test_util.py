from dataclasses import dataclass
from pathlib import Path

from torch_playground.util import parse_cmd_line_args, BaseArguments

class TestUtil:
    def test_parse_cmd_line_args_base(self):
        """Test parsing command line arguments with no arguments."""
        args = parse_cmd_line_args(BaseArguments(), description=None, argv=[])
        assert isinstance(args, BaseArguments)
        # BaseArguments provides shared logging fields.
        assert len(vars(args)) == 2, 'Expected two arguments to be parsed.'
        assert hasattr(args, 'loglevel')
        assert args.loglevel == 'INFO', 'Default log level should be INFO.'
        assert hasattr(args, 'logdir')
        assert args.logdir == BaseArguments.logdir, 'Default log directory should match the expected path.'

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
