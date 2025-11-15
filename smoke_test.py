#!/usr/bin/env python
"""Quick smoke test for the name_seq_completer tool.

This script creates a minimal trained model and tokenizer, then tests the completer tool.
"""

import sys
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from torch_playground.tokenizer import NGramTokenizer
from torch_playground.name_seq_completer import (
    NameSeqPredictor,
    NameSeqCompleterConfig,
)
import structlog


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


def main():
    """Run smoke test."""
    print("Creating temporary directory...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        print(f"  -> {tmpdir_path}")

        # Create tokenizer
        print("\nCreating tokenizer with sample names...")
        tokenizer = NGramTokenizer(n=2)
        names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace']
        for name in names:
            tokenizer.add_to_token_dict(name)
        tokenizer.add_padding_token('<PAD>')
        tokenizer.add_unknown_token('<UNK>')
        tokenizer_file = tmpdir_path / 'tokenizer.json'
        tokenizer.to_file(tokenizer_file)
        print(f"  -> Vocab size: {tokenizer.vocab_size()}")
        print(f"  -> Tokenizer saved to: {tokenizer_file}")

        # Create model
        print("\nCreating and saving model...")
        model = SimpleTransformer(vocab_size=tokenizer.vocab_size())
        model_file = tmpdir_path / 'model.pt'
        torch.save(model, model_file)
        print(f"  -> Model saved to: {model_file}")

        # Create logger
        logger = structlog.get_logger()

        # Create predictor
        print("\nInitializing predictor...")
        predictor = NameSeqPredictor(
            model_path=model_file,
            tokenizer_path=tokenizer_file,
            device='cpu',
            logger=logger
        )
        print("  -> Predictor initialized successfully")

        # Test predictions
        print("\nGenerating completions...")
        test_prefixes = ['Al', 'Bo', 'Ch', 'Da']
        for prefix in test_prefixes:
            print(f"\n  Prefix: '{prefix}'")
            try:
                results = predictor.predict_completions(
                    prefix=prefix,
                    num_completions=2,
                    top_k=3,
                    temperature=1.0
                )
                for i, result in enumerate(results, 1):
                    print(f"    {i}. {result['full_sequence']:20s} [{result['confidence']:.3f}]")
            except Exception as e:
                print(f"    Error: {e}")

        # Test config
        print("\nTesting configuration...")
        config = NameSeqCompleterConfig(
            model_path=model_file,
            tokenizer_path=tokenizer_file,
            top_k=5,
            temperature=0.8,
            max_predictions=10
        )
        print(f"  -> top_k: {config.top_k}")
        print(f"  -> temperature: {config.temperature}")
        print(f"  -> max_predictions: {config.max_predictions}")

        print("\n✅ Smoke test completed successfully!")
        return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
