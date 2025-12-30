"""Tests for batch mode functionality of the name sequence completer."""

import json
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import structlog

from torch_playground.tokenizer import NGramTokenizer
from torch_playground.name_seq_completer import (
    NameSeqCompleterConfig,
    NameSeqPredictor,
    NameSeqCompleterApp,
)


class SimpleTransformer(nn.Module):
    """Minimal transformer for testing."""

    def __init__(self, vocab_size: int = 100, d_model: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            num_encoder_layers=1,
            num_decoder_layers=1,
            nhead=2,
            dim_feedforward=128,
            batch_first=True,
        )
        self.logits_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        xformer_out = self.transformer(src=src_emb, tgt=tgt_emb)
        logits = self.logits_layer(xformer_out)
        return logits


@pytest.fixture
def tokenizer():
    """Create a test tokenizer."""
    tokenizer = NGramTokenizer(n=1)
    names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
    for name in names:
        tokenizer.add_to_token_dict(name)
    tokenizer.add_padding_token('<PAD>')
    tokenizer.add_unknown_token('<UNK>')
    return tokenizer


@pytest.fixture
def model():
    """Create a test model."""
    return SimpleTransformer(vocab_size=100, d_model=32)


@pytest.fixture
def temp_files(tokenizer, model):
    """Create temporary model and tokenizer files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save tokenizer
        tokenizer_file = tmpdir / 'tokenizer.json'
        tokenizer.to_file(tokenizer_file)

        # Save model
        model_file = tmpdir / 'model.pt'
        torch.save(model, model_file)

        yield {
            'tokenizer': tokenizer_file,
            'model': model_file,
            'tmpdir': tmpdir
        }


class TestBatchMode:
    """Test batch mode processing."""

    def test_batch_mode_with_valid_input(self, temp_files):
        """Test batch mode with valid input file."""
        # Create input file
        input_file = temp_files['tmpdir'] / 'input.txt'
        prefixes = ['Al', 'Bo', 'Ch']
        with open(input_file, 'w') as f:
            f.write('\n'.join(prefixes))

        output_file = temp_files['tmpdir'] / 'output.json'

        # Create app with CLI arguments
        argv = [
            '--model_path', str(temp_files['model']),
            '--tokenizer_path', str(temp_files['tokenizer']),
            '--mode', 'batch',
            '--input_file', str(input_file),
            '--output_file', str(output_file),
            '--max_predictions', '3',
            '--top_k', '2',
            '--temperature', '1.0',
        ]
        app = NameSeqCompleterApp(argv=argv)

        # Run batch mode
        app.run()

        # Verify output file exists and contains valid JSON
        assert output_file.exists()

        with open(output_file) as f:
            results = json.load(f)

        # Verify structure
        assert isinstance(results, list)
        assert len(results) == len(prefixes)

        for i, result in enumerate(results):
            assert 'prefix' in result
            assert result['prefix'] == prefixes[i]
            assert 'completions' in result
            assert isinstance(result['completions'], list)

            # Each completion should have required fields
            for completion in result['completions']:
                assert 'full_sequence' in completion
                assert 'confidence' in completion
                assert 'rank' in completion
                assert 0 <= completion['confidence'] <= 1
                assert completion['rank'] >= 1

    def test_batch_mode_missing_input_file(self, temp_files):
        """Test batch mode with missing input file."""
        missing_input = temp_files['tmpdir'] / 'nonexistent.txt'
        output_file = temp_files['tmpdir'] / 'output.json'

        argv = [
            '--model_path', str(temp_files['model']),
            '--tokenizer_path', str(temp_files['tokenizer']),
            '--mode', 'batch',
            '--input_file', str(missing_input),
            '--output_file', str(output_file),
        ]
        app = NameSeqCompleterApp(argv=argv)

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            app.run()

    def test_batch_mode_missing_input_file_config(self, temp_files):
        """Test batch mode without input_file in config."""
        output_file = temp_files['tmpdir'] / 'output.json'

        argv = [
            '--model_path', str(temp_files['model']),
            '--tokenizer_path', str(temp_files['tokenizer']),
            '--mode', 'batch',
            '--output_file', str(output_file),
        ]
        app = NameSeqCompleterApp(argv=argv)

        # Should raise ValueError
        with pytest.raises(ValueError, match='input_file required'):
            app.run()

    def test_batch_mode_missing_output_file(self, temp_files):
        """Test batch mode without output_file in config."""
        # Create input file
        input_file = temp_files['tmpdir'] / 'input.txt'
        with open(input_file, 'w') as f:
            f.write('Al\nBo')

        argv = [
            '--model_path', str(temp_files['model']),
            '--tokenizer_path', str(temp_files['tokenizer']),
            '--mode', 'batch',
            '--input_file', str(input_file),
        ]
        app = NameSeqCompleterApp(argv=argv)

        # Should raise ValueError
        with pytest.raises(ValueError, match='output_file required'):
            app.run()

    def test_batch_mode_empty_input_file(self, temp_files):
        """Test batch mode with empty input file."""
        # Create empty input file
        input_file = temp_files['tmpdir'] / 'empty.txt'
        input_file.touch()

        output_file = temp_files['tmpdir'] / 'output.json'

        argv = [
            '--model_path', str(temp_files['model']),
            '--tokenizer_path', str(temp_files['tokenizer']),
            '--mode', 'batch',
            '--input_file', str(input_file),
            '--output_file', str(output_file),
        ]
        app = NameSeqCompleterApp(argv=argv)

        app.run()

        # Verify output is valid but empty
        with open(output_file) as f:
            results = json.load(f)

        assert isinstance(results, list)
        assert len(results) == 0

    def test_batch_mode_with_invalid_prefix(self, temp_files):
        """Test batch mode with invalid prefix that can't be tokenized."""
        # Create input file with prefix that's not in tokenizer
        input_file = temp_files['tmpdir'] / 'input.txt'
        with open(input_file, 'w') as f:
            f.write('ZZZZZZZZZ\n')  # Not in tokenizer vocabulary

        output_file = temp_files['tmpdir'] / 'output.json'

        argv = [
            '--model_path', str(temp_files['model']),
            '--tokenizer_path', str(temp_files['tokenizer']),
            '--mode', 'batch',
            '--input_file', str(input_file),
            '--output_file', str(output_file),
        ]
        app = NameSeqCompleterApp(argv=argv)

        app.run()

        # Should still produce output with error fields
        with open(output_file) as f:
            results = json.load(f)

        assert len(results) == 1
        assert results[0]['prefix'] == 'ZZZZZZZZZ'
        assert 'error' in results[0]

    def test_batch_mode_preserves_whitespace_in_input(self, temp_files):
        """Test that batch mode preserves prefix formatting."""
        # Create input file with various whitespace
        input_file = temp_files['tmpdir'] / 'input.txt'
        with open(input_file, 'w') as f:
            f.write('Al\n\nBo\n  \nCh\n')  # Mixed with blank lines and spaces

        output_file = temp_files['tmpdir'] / 'output.json'

        argv = [
            '--model_path', str(temp_files['model']),
            '--tokenizer_path', str(temp_files['tokenizer']),
            '--mode', 'batch',
            '--input_file', str(input_file),
            '--output_file', str(output_file),
            '--max_predictions', '1',
        ]
        app = NameSeqCompleterApp(argv=argv)

        app.run()

        with open(output_file) as f:
            results = json.load(f)

        # Should have 3 results (blank lines stripped)
        assert len(results) == 3
        prefixes = [r['prefix'] for r in results]
        assert prefixes == ['Al', 'Bo', 'Ch']

    def test_batch_mode_output_format(self, temp_files):
        """Test batch mode output JSON format."""
        input_file = temp_files['tmpdir'] / 'input.txt'
        with open(input_file, 'w') as f:
            f.write('Al')

        output_file = temp_files['tmpdir'] / 'output.json'

        argv = [
            '--model_path', str(temp_files['model']),
            '--tokenizer_path', str(temp_files['tokenizer']),
            '--mode', 'batch',
            '--input_file', str(input_file),
            '--output_file', str(output_file),
            '--max_predictions', '2',
            '--top_k', '2',
        ]
        app = NameSeqCompleterApp(argv=argv)

        app.run()

        with open(output_file) as f:
            results = json.load(f)

        # Verify exact output structure
        assert len(results) >= 1
        result = results[0]

        # Required top-level keys
        assert set(result.keys()) >= {'prefix', 'completions'}

        # Verify completion ranking
        completions = result['completions']
        if len(completions) > 1:
            # Confidence should be in decreasing order
            confidences = [c['confidence'] for c in completions]
            assert all(
                confidences[i] >= confidences[i+1]
                for i in range(len(confidences) - 1)
            )


class TestBatchModeIntegration:
    """Integration tests for batch mode with real components."""

    def test_batch_mode_end_to_end(self, temp_files):
        """Test complete batch processing pipeline."""
        # Create input with multiple prefixes
        input_file = temp_files['tmpdir'] / 'names.txt'
        test_prefixes = ['A', 'B', 'C', 'D', 'E']
        with open(input_file, 'w') as f:
            f.write('\n'.join(test_prefixes))

        output_file = temp_files['tmpdir'] / 'completions.json'

        # Run batch mode
        argv = [
            '--model_path', str(temp_files['model']),
            '--tokenizer_path', str(temp_files['tokenizer']),
            '--mode', 'batch',
            '--input_file', str(input_file),
            '--output_file', str(output_file),
            '--max_predictions', '5',
            '--top_k', '3',
            '--temperature', '0.9',
        ]
        app = NameSeqCompleterApp(argv=argv)
        app.run()

        # Verify all prefixes were processed
        with open(output_file) as f:
            results = json.load(f)

        assert len(results) == len(test_prefixes)

        # Verify each result is properly formatted
        for result in results:
            if 'error' not in result:
                # Valid result
                assert 'prefix' in result
                assert 'completions' in result
                assert isinstance(result['completions'], list)
                assert len(result['completions']) > 0

    def test_batch_mode_reproducibility(self, temp_files):
        """Test that batch mode produces consistent results with fixed seed."""
        input_file = temp_files['tmpdir'] / 'input.txt'
        with open(input_file, 'w') as f:
            f.write('Al\nBo')

        output_file1 = temp_files['tmpdir'] / 'output1.json'
        output_file2 = temp_files['tmpdir'] / 'output2.json'

        # Run batch mode twice with same seed and temperature
        for output_file in [output_file1, output_file2]:
            argv = [
                '--model_path', str(temp_files['model']),
                '--tokenizer_path', str(temp_files['tokenizer']),
                '--mode', 'batch',
                '--input_file', str(input_file),
                '--output_file', str(output_file),
                '--randseed', '42',
                '--temperature', '0.0',  # Deterministic: always pick max
            ]
            app = NameSeqCompleterApp(argv=argv)
            app.run()

        # Compare outputs
        with open(output_file1) as f:
            results1 = json.load(f)
        with open(output_file2) as f:
            results2 = json.load(f)

        # Results should be identical (same order and completions)
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1['prefix'] == r2['prefix']
            assert len(r1['completions']) == len(r2['completions'])
