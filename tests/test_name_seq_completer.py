"""Unit tests for name_seq_completer components."""

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json

from torch_playground.name_seq_completer import (
    NameSeqCompleterConfig,
    NameSeqPredictor,
    NameSeqCompleterApp,
    InteractiveShell,
)
from torch_playground.tokenizer import NGramTokenizer


class TestNameSeqCompleterConfig:
    """Tests for NameSeqCompleterConfig."""

    def test_config_initialization_defaults(self):
        """Test config initialization with defaults."""
        config = NameSeqCompleterConfig()
        assert config.max_predictions == 10
        assert config.top_k == 5
        assert config.temperature == 1.0
        assert config.mode == 'interactive'

    def test_config_initialization_custom(self):
        """Test config initialization with custom values."""
        config = NameSeqCompleterConfig(
            max_predictions=20,
            top_k=3,
            temperature=0.8,
            mode='batch'
        )
        assert config.max_predictions == 20
        assert config.top_k == 3
        assert config.temperature == 0.8
        assert config.mode == 'batch'

    def test_config_required_paths(self):
        """Test that model_path and tokenizer_path are required (defaults to /dev/null)."""
        config = NameSeqCompleterConfig()
        assert config.model_path == Path('/dev/null')
        assert config.tokenizer_path == Path('/dev/null')


class TestNameSeqPredictor:
    """Tests for NameSeqPredictor."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock(spec=NGramTokenizer)
        tokenizer.vocab_size.return_value = 100
        tokenizer.tokenize.return_value = [0, 1, 2]
        tokenizer.decode.return_value = 'decoded_text'
        tokenizer.get_token_string.side_effect = lambda x: f'token_{x}'
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.eval = Mock(return_value=None)
        # Model returns logits of shape (batch, seq_len, vocab_size)
        model.return_value = torch.randn(1, 3, 100)
        return model

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        logger = Mock()
        logger.bind.return_value = logger
        return logger

    def test_predictor_initialization(self, mock_tokenizer, mock_model, mock_logger, tmp_path):
        """Test predictor initialization."""
        with patch('torch_playground.name_seq_completer.torch.load', return_value=mock_model):
            with patch('torch_playground.name_seq_completer.NGramTokenizer.from_file', return_value=mock_tokenizer):
                predictor = NameSeqPredictor(
                    model_path=tmp_path / 'model.pt',
                    tokenizer_path=tmp_path / 'tokenizer.json',
                    device='cpu',
                    logger=mock_logger
                )
                assert predictor.model is not None
                assert predictor.tokenizer is not None
                assert predictor.device == 'cpu'

    def test_predictor_encode_prefix(self, mock_tokenizer, mock_model, mock_logger, tmp_path):
        """Test prefix encoding."""
        with patch('torch_playground.name_seq_completer.torch.load', return_value=mock_model):
            with patch('torch_playground.name_seq_completer.NGramTokenizer.from_file', return_value=mock_tokenizer):
                predictor = NameSeqPredictor(
                    model_path=tmp_path / 'model.pt',
                    tokenizer_path=tmp_path / 'tokenizer.json',
                    device='cpu',
                    logger=mock_logger
                )
                result = predictor._encode_prefix('hello')
                assert result == [0, 1, 2]
                mock_tokenizer.tokenize.assert_called_with('hello')

    def test_predictor_encode_prefix_invalid(self, mock_tokenizer, mock_model, mock_logger, tmp_path):
        """Test error handling for invalid prefix."""
        mock_tokenizer.tokenize.side_effect = KeyError('unknown token')
        with patch('torch_playground.name_seq_completer.torch.load', return_value=mock_model):
            with patch('torch_playground.name_seq_completer.NGramTokenizer.from_file', return_value=mock_tokenizer):
                predictor = NameSeqPredictor(
                    model_path=tmp_path / 'model.pt',
                    tokenizer_path=tmp_path / 'tokenizer.json',
                    device='cpu',
                    logger=mock_logger
                )
                with pytest.raises(ValueError):
                    predictor._encode_prefix('unknown_chars_xyz')

    def test_get_next_token_probabilities(self, mock_tokenizer, mock_model, mock_logger, tmp_path):
        """Test getting next token probabilities."""
        mock_model.return_value = torch.randn(1, 3, 100)
        with patch('torch_playground.name_seq_completer.torch.load', return_value=mock_model):
            with patch('torch_playground.name_seq_completer.NGramTokenizer.from_file', return_value=mock_tokenizer):
                predictor = NameSeqPredictor(
                    model_path=tmp_path / 'model.pt',
                    tokenizer_path=tmp_path / 'tokenizer.json',
                    device='cpu',
                    logger=mock_logger
                )
                logits = predictor._get_next_token_probabilities([0, 1, 2])
                assert logits.shape == (100,)


class TestInteractiveShell:
    """Tests for InteractiveShell."""

    @pytest.fixture
    def mock_predictor(self):
        """Create a mock predictor."""
        predictor = Mock(spec=NameSeqPredictor)
        predictor.predict_completions.return_value = [
            {
                'completion': 'na',
                'full_sequence': 'Johna',
                'confidence': 0.92,
                'token_ids': [10, 20],
                'probabilities': [0.92, 0.85],
            }
        ]
        return predictor

    @pytest.fixture
    def shell(self, mock_predictor):
        """Create an interactive shell instance."""
        config = NameSeqCompleterConfig()
        logger = Mock()
        logger.bind.return_value = logger
        return InteractiveShell(
            predictor=mock_predictor,
            config=config,
            logger=logger
        )

    def test_shell_initialization(self, shell):
        """Test shell initialization."""
        assert shell.predictor is not None
        assert shell.config is not None
        assert shell.prompt == '(NameSeqCompleter) '

    def test_shell_do_config(self, shell, capsys):
        """Test config command output."""
        shell.do_config('')
        captured = capsys.readouterr()
        assert 'Current Configuration:' in captured.out
        assert 'top_k:' in captured.out

    def test_shell_do_topk_valid(self, shell, capsys):
        """Test topk command with valid input."""
        shell.do_topk('3')
        assert shell.config.top_k == 3
        captured = capsys.readouterr()
        assert 'Updated top_k to 3' in captured.out

    def test_shell_do_topk_invalid(self, shell, capsys):
        """Test topk command with invalid input."""
        original_k = shell.config.top_k
        shell.do_topk('invalid')
        assert shell.config.top_k == original_k
        captured = capsys.readouterr()
        assert 'Error' in captured.out

    def test_shell_do_topk_negative(self, shell, capsys):
        """Test topk command with negative input."""
        original_k = shell.config.top_k
        shell.do_topk('-1')
        assert shell.config.top_k == original_k
        captured = capsys.readouterr()
        assert 'Error' in captured.out

    def test_shell_do_topk_display_current(self, shell, capsys):
        """Test topk command displays current value when no arg."""
        shell.do_topk('')
        captured = capsys.readouterr()
        assert 'Current top_k:' in captured.out

    def test_shell_do_complete(self, shell, capsys):
        """Test complete command."""
        shell.do_complete('John')
        captured = capsys.readouterr()
        assert 'Completions for "John":' in captured.out

    def test_shell_do_complete_no_args(self, shell, capsys):
        """Test complete command with no arguments."""
        shell.do_complete('')
        captured = capsys.readouterr()
        assert 'Usage:' in captured.out

    def test_shell_default_empty_input(self, shell, capsys):
        """Test default handler with empty input."""
        shell.default('   ')
        # Should silently ignore empty input
        captured = capsys.readouterr()
        assert captured.out == ''

    def test_shell_help_complete(self, shell, capsys):
        """Test help text for complete command."""
        shell.help_complete()
        captured = capsys.readouterr()
        assert 'complete' in captured.out.lower()

    def test_shell_help_config(self, shell, capsys):
        """Test help text for config command."""
        shell.help_config()
        captured = capsys.readouterr()
        assert 'config' in captured.out.lower()

    def test_shell_help_topk(self, shell, capsys):
        """Test help text for topk command."""
        shell.help_topk()
        captured = capsys.readouterr()
        assert 'topk' in captured.out.lower()


class TestNameSeqCompleterApp:
    """Tests for NameSeqCompleterApp."""

    def test_app_initialization(self):
        """Test app initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app = NameSeqCompleterApp(argv=[
                '--model_path', '/tmp/model.pt',
                '--tokenizer_path', '/tmp/tokenizer.json'
            ])
            assert app.config is not None
            assert app.config.model_path == Path('/tmp/model.pt')

    def test_app_mode_batch(self):
        """Test app recognizes batch mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app = NameSeqCompleterApp(argv=[
                '--model_path', '/tmp/model.pt',
                '--tokenizer_path', '/tmp/tokenizer.json',
                '--mode', 'batch'
            ])
            assert app.config.mode == 'batch'

    def test_app_mode_interactive(self):
        """Test app recognizes interactive mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            app = NameSeqCompleterApp(argv=[
                '--model_path', '/tmp/model.pt',
                '--tokenizer_path', '/tmp/tokenizer.json',
                '--mode', 'interactive'
            ])
            assert app.config.mode == 'interactive'
