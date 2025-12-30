"""Integration test for the name sequence completer tool."""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from torch_playground.tokenizer import NGramTokenizer
from torch_playground.name_seq_completer import (
    NameSeqCompleterConfig,
    NameSeqPredictor,
    NameSeqCompleterApp,
)


class SimpleTransformer(torch.nn.Module):
    """Minimal transformer for testing."""

    def __init__(self, vocab_size: int = 100, d_model: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            num_encoder_layers=1,
            num_decoder_layers=1,
            nhead=2,
            dim_feedforward=128,
            batch_first=True,
        )
        self.logits_layer = torch.nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        xformer_out = self.transformer(src=src_emb, tgt=tgt_emb)
        logits = self.logits_layer(xformer_out)
        return logits


def test_integration_predictor_with_real_tokenizer():
    """Test NameSeqPredictor with a real tokenizer and mock model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create and save tokenizer
        tokenizer = NGramTokenizer(n=2)
        tokenizer.add_to_token_dict('hello')
        tokenizer.add_padding_token('<PAD>')
        tokenizer_file = tmpdir_path / 'tokenizer.json'
        tokenizer.to_file(tokenizer_file)

        # Create and save model
        model = SimpleTransformer(vocab_size=tokenizer.vocab_size())
        model_file = tmpdir_path / 'model.pt'
        torch.save(model, model_file)

        # Create mock logger
        import structlog
        logger = structlog.get_logger()

        # Create predictor
        predictor = NameSeqPredictor(
            model_path=model_file,
            tokenizer_path=tokenizer_file,
            device='cpu',
            logger=logger
        )

        # Test encoding
        prefix_tokens = predictor._encode_prefix('he')
        assert len(prefix_tokens) > 0

        # Test next token probabilities
        probs = predictor._get_next_token_probabilities(prefix_tokens)
        assert probs.shape[0] == tokenizer.vocab_size()

        # Test predictions
        results = predictor.predict_completions(
            prefix='he',
            num_completions=2,
            top_k=3,
            temperature=1.0
        )
        assert len(results) > 0
        assert 'completion' in results[0]
        assert 'confidence' in results[0]
        assert 'full_sequence' in results[0]


def test_integration_app_initialization():
    """Test NameSeqCompleterApp initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create dummy files
        model_file = tmpdir_path / 'model.pt'
        model_file.touch()
        tokenizer_file = tmpdir_path / 'tokenizer.json'
        tokenizer_file.touch()

        # Test initialization with argv
        app = NameSeqCompleterApp(argv=[
            '--model_path', str(model_file),
            '--tokenizer_path', str(tokenizer_file),
            '--top_k', '3',
            '--temperature', '0.8',
        ])

        assert app.config.model_path == model_file
        assert app.config.tokenizer_path == tokenizer_file
        assert app.config.top_k == 3
        assert app.config.temperature == 0.8


def test_integration_tokenizer_decode_roundtrip():
    """Test that tokenizer decode works end-to-end."""
    # Create tokenizer with single-char tokens
    tokenizer = NGramTokenizer(n=1)
    text = 'hello'
    tokenizer.add_to_token_dict(text)

    # Encode
    token_ids = tokenizer.tokenize(text)
    assert len(token_ids) == len(text)

    # Decode
    decoded = tokenizer.decode(token_ids)
    assert decoded == text

    # Test with tensor
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    decoded_from_tensor = tokenizer.decode(token_tensor)
    assert decoded_from_tensor == text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
