from dataclasses import dataclass, fields, field
from pathlib import Path
import pytest
import torch
import re
import json
from typing import Iterator

from torch_playground.util import (
    parse_cmd_line_args,
    BaseConfiguration,
    setup_logging,
    accuracy,
    TrainableModelApp,
    save_tensor,
    SequenceCrossEntropyLoss,
    BaseApp,
    FileDataset,
    TransformableMixin,
)


class MinimalApp[BaseConfiguration](BaseApp):
    """A minimal app that implements the run() method as a no-op, for pure testing purposes."""
    def run(self):
        pass


class SimpleIterator:
    """A simple iterator class for testing the TransformableMixin."""
    def __init__(self, data: list):
        self.data = data

    def __iter__(self) -> Iterator:
        return self._apply_transforms(iter(self.data)) if hasattr(self, '_apply_transforms') else iter(self.data)


class TransformableIterator(TransformableMixin, SimpleIterator):
    """Test class that combines TransformableMixin with a simple iterator."""
    def __init__(self, data: list):
        TransformableMixin.__init__(self)
        SimpleIterator.__init__(self, data)


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
        with pytest.raises(TypeError):
            # Deliberately try to violate the ABC barrier - should fail. Pylance recognizes
            # this as a static error, so disable that check for this test.
            app = BaseApp(BaseConfiguration(), description='Test failing app', argv=[]) # pyright: ignore[reportAbstractUsage]
            app.run()

    def test_file_dataset_basic_iteration(self, tmp_path: Path):
        """Test that FileDataset correctly iterates over file contents without transforms."""
        test_file = tmp_path / "test.txt"
        test_content = "line1\nline2\nline3\n"
        test_file.write_text(test_content)

        dataset = FileDataset(test_file)
        lines = list(dataset)
        assert lines == ["line1\n", "line2\n", "line3\n"]

    def test_file_dataset_single_transform(self, tmp_path: Path):
        """Test that FileDataset correctly applies a single transform."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("1\n2\n3\n")

        dataset = FileDataset(test_file)
        def to_int(x: str) -> int:
            return int(x.strip())
        dataset.with_transform(to_int)
        numbers = list(dataset)
        assert numbers == [1, 2, 3]

    def test_file_dataset_multiple_transforms(self, tmp_path: Path):
        """Test that FileDataset correctly composes multiple transforms."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("1\n2\n3\n")

        dataset = FileDataset(test_file)
        def to_int(x: str) -> int:
            return int(x.strip())
        def double(x: int) -> int:
            return x * 2
        def format_num(x: int) -> str:
            return f"Number: {x}"

        (dataset.with_transform(to_int)        # string -> int
                .with_transform(double)        # int -> int*2
                .with_transform(format_num))    # int -> formatted string

        results = list(dataset)
        assert results == ["Number: 2", "Number: 4", "Number: 6"]

    def test_file_dataset_transform_order(self, tmp_path: Path):
        """Test that transforms are applied in the correct order (first to last)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("5\n")

        dataset = FileDataset(test_file)
        def to_int(x: str) -> int:
            return int(x.strip())
        def add_three(x: int) -> int:
            return x + 3
        def to_str(x: int) -> str:
            return str(x)
        def wrap_parens(x: str) -> str:
            return f"({x})"

        # These transforms should be applied in order. If applied in reverse order,
        # we would get a different result
        dataset.with_transform(to_int)       # "5\n" -> 5
        dataset.with_transform(add_three)    # 5 -> 8
        dataset.with_transform(to_str)       # 8 -> "8"
        dataset.with_transform(wrap_parens)  # "8" -> "(8)"

        result = next(iter(dataset))
        assert result == "(8)"  # If transforms were applied in reverse order, we would get "(5)3"

    def test_file_dataset_chaining(self, tmp_path: Path):
        """Test that with_transform method supports method chaining."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test\n")

        dataset = FileDataset(test_file)
        def strip_text(x: str) -> str:
            return x.strip()
        # Should return self for chaining
        result = dataset.with_transform(strip_text)
        assert result is dataset

    def test_transformable_mixin_basic(self):
        """Test basic transform operations on a simple iterator."""
        # Setup
        data = ['1', '2', '3']
        iterator = TransformableIterator(data)

        # Test no transforms
        result = list(iterator)
        assert result == data, 'Iterator should return original data when no transforms are applied'

        # Test single transform
        iterator = TransformableIterator(data)
        iterator.with_transform(int)
        result = list(iterator)
        assert result == [1, 2, 3], 'Single transform should convert strings to integers'

        # Test transform chain
        iterator = TransformableIterator(data)
        iterator.with_transform(int).with_transform(lambda x: x * 2)
        result = list(iterator)
        assert result == [2, 4, 6], 'Transform chain should apply transforms in order'

    def test_transform_empty_data(self):
        """Test transforms on empty data."""
        iterator = TransformableIterator([])
        iterator.with_transform(int)
        result = list(iterator)
        assert result == [], 'Empty iterator should remain empty after transforms'

    def test_transform_invalid_input(self):
        """Test transform chain with invalid input."""
        data = ['1', 'a', '3']
        iterator = TransformableIterator(data)
        iterator.with_transform(int)
        with pytest.raises(ValueError):
            list(iterator)

    def test_file_dataset_with_transforms(self, tmp_path: Path):
        """Test transforms with FileDataset."""
        # Create a temporary file with test data
        test_file = tmp_path / 'test.txt'
        test_file.write_text('1\n2\n3\n')

        # Test transforming lines to integers
        dataset = FileDataset(test_file)
        # First strip the newline from each line, then convert to int
        dataset.with_transform(str.strip).with_transform(int)
        result = list(dataset)
        assert result == [1, 2, 3], 'Should convert file lines to integers'

    def test_transform_method_chaining(self):
        """Test that transform method chaining works correctly."""
        data = ['1', '2', '3']
        iterator = TransformableIterator(data)

        # Test that with_transform returns self
        assert iterator.with_transform(int) is iterator, 'with_transform should return self'

        # Add transforms via chaining
        iterator.with_transform(lambda x: x * 2).with_transform(lambda x: f"Number: {x}")

        # Now test the result
        result = list(iterator)
        assert result == ['Number: 2', 'Number: 4', 'Number: 6'], 'Chained transforms should work'

    def test_transform_complex_types(self):
        """Test transforms with more complex data types and transformations."""
        data = ['{"a": 1}', '{"b": 2}', '{"c": 3}']
        iterator = TransformableIterator(data)

        import json
        def extract_value(d: dict) -> int:
            return list(d.values())[0]

        iterator.with_transform(json.loads).with_transform(extract_value)
        result = list(iterator)
        assert result == [1, 2, 3], 'Should handle complex transforms with multiple types'

    def test_file_dataset_large_file(self, tmp_path: Path):
        """Test FileDataset with transforms on a larger file."""
        # Create a temporary file with more test data
        test_file = tmp_path / 'large.txt'
        test_file.write_text('\n'.join([str(x) for x in range(1000)]))
        # with test_file.open('w') as f:
        #     for i in range(1000):
        #         f.write(f'{i}\n')

        # Test processing with transforms
        dataset = FileDataset(test_file)
        dataset.with_transform(str.strip).with_transform(int).with_transform(lambda x: x * 2)

        # Check first few and last few items
        result = list(dataset)
        assert len(result) == 1000, 'Should process all lines'
        assert result[:3] == [0, 2, 4], 'First three items should be transformed correctly'
        assert result[-3:] == [1994, 1996, 1998], 'Last three items should be transformed correctly'

    def test_save_tensor_simple_tensor(self, tmp_path: Path):
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

    def test_app_init_saves_config(self, tmp_path: Path):
        @dataclass
        class LocalConfig(BaseConfiguration):
            foo: int = field(default=7)
            bar: str = field(default='Twas brillig and the slithy toves')
        app = MinimalApp(LocalConfig(), description='testing app', argv=['--output_dir', str(tmp_path),
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