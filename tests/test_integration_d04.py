import pytest
from pathlib import Path
from torch_playground.d04_name_seq_learner import NameSeqLearnerApp
from torch_playground.tokenizer import NGramTokenizer


@pytest.mark.integration
class TestIntegrationD04NameSeqLearner:

    @pytest.mark.parametrize('ngram_length', [1, 2])
    def test_full_run(self, tmp_path: Path, ngram_length: int):
        # Create a small names file with many entries so splits are non-empty
        names = [f'name{i}' for i in range(30)]
        names_file = tmp_path / 'names.txt'
        names_file.write_text('\n'.join(names))

        # Build a minimal tokenizer and save it
        tokenizer = NGramTokenizer(n=ngram_length)
        tokenizer.add_unknown_token('<UNKNOWN:_>')
        tokenizer.add_padding_token('<PAD>')
        for n in names:
            tokenizer.add_to_token_dict(n)
        tokenizer_file = tmp_path / f'token_dict.n={ngram_length}.json'
        tokenizer.to_file(tokenizer_file)

        # Run the app with small model and training settings
        app = NameSeqLearnerApp([
            '--names_file', str(names_file),
            '--tokenizer_file', str(tokenizer_file),
            '--epochs', '2',
            '--batch_size', '4',
            '--output_dir', str(tmp_path / 'output'),
            '--d_embedding', '8',
            '--d_feedforward', '16',
            '--n_heads', '2',
            '--n_encoder_layers', '1',
            '--n_decoder_layers', '1',
            '--in_seq_length', '8',
            '--out_seq_length', '8',
        ])
        app.run()

        expected_files = [
            'config.json',
            'model_summary.txt',
            'trained_model.pt',
            'train_indices.txt',
            'test_indices.txt',
            'val_indices.txt',
        ]
        out_dir = app.work_dir
        assert tmp_path in out_dir.parents
        for e in expected_files:
            assert (out_dir / e).exists()
