import pytest
from dataclasses import dataclass

from torch_playground.util import parse_cmd_line_args, BaseArguments

class TestUtil:
    def test_parse_cmd_line_args_empty(self):
        """Test parsing command line arguments with no arguments."""
        args = parse_cmd_line_args(BaseArguments(), [])
        assert isinstance(args, BaseArguments)
        assert len(vars(args)) == 0, "Expected no arguments to be parsed."

    def test_parse_cmd_line_args_with_args_defaults(self):
        """Test parsing command line arguments defaulted to provided vals."""
        @dataclass
        class CustomArgs(BaseArguments):
            arg1: str = 'default1'
            arg2: int = 42
        args = parse_cmd_line_args(CustomArgs(), [])
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
        args = parse_cmd_line_args(CustomArgs2(), ['--arg21', 'value1', '--arg22', '100'])
        assert isinstance(args, CustomArgs2)
        assert hasattr(args, 'arg21')
        assert args.arg21 == 'value1'
        assert hasattr(args, 'arg22')
        assert args.arg22 == 100
