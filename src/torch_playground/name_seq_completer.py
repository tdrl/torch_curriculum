"""Interactive tool for completing name sequences using trained transformer models."""

import cmd
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import structlog

from torch_playground.util import BaseConfiguration, BaseApp, select_device
from torch_playground.tokenizer import NGramTokenizer
from torch_playground.d04_name_seq_learner import NameSeqTransformer, NameSeqLearnerConfig


__all__ = [
    'NameSeqCompleterConfig',
    'NameSeqPredictor',
    'NameSeqCompleterApp',
    'InteractiveShell',
]


@dataclass
class NameSeqCompleterConfig(BaseConfiguration):
    """Configuration for the Name Sequence Completer tool."""

    # Model/Tokenizer I/O
    model_path: Path = field(
        default=Path('/dev/null'),
        metadata=BaseConfiguration._meta(
            help='Path to trained model (.pt file)',
            required=True
        )
    )
    config_path: Path = field(
        default=Path('/dev/null'),
        metadata=BaseConfiguration._meta(
            help='Path to model config file (.json). Must match the config used to train the model.',
            required=True
        )
    )
    tokenizer_path: Path = field(
        default=Path('/dev/null'),
        metadata=BaseConfiguration._meta(
            help='Path to tokenizer file (.json)',
            required=True
        )
    )

    # Prediction parameters
    max_predictions: int = field(
        default=10,
        metadata=BaseConfiguration._meta(help='Number of completions to generate per prefix.')
    )
    top_k: int = field(
        default=5,
        metadata=BaseConfiguration._meta(help='Number of top predictions to display.')
    )
    temperature: float = field(
        default=1.0,
        metadata=BaseConfiguration._meta(help='Temperature for softmax probability scaling (lower=sharper).')
    )

    # Mode selection
    mode: str = field(
        default='interactive',
        metadata=BaseConfiguration._meta(help='Mode: "interactive" or "batch".')
    )
    input_file: str | None = field(
        default=None,
        metadata=BaseConfiguration._meta(help='For batch mode: input file with prefixes (one per line).')
    )
    output_file: str | None = field(
        default=None,
        metadata=BaseConfiguration._meta(help='For batch mode: output file for predictions (JSON format).')
    )


class NameSeqPredictor:
    """Wrapper around NameSeqTransformer for completion predictions.

    Responsibilities:
    - Load model and tokenizer
    - Handle prefix encoding
    - Generate predictions using the model
    - Score predictions (probabilities/confidence)
    - Decode token IDs back to strings
    """

    def __init__(self,
                 model_path: Path,
                 config_path: Path,
                 tokenizer_path: Path,
                 device: str,
                 logger: structlog.BoundLogger):
        """Initialize the predictor with model and tokenizer.

        Args:
            model_path: Path to trained PyTorch model (.pt file)
            config_path: Path to model configuration JSON file
            tokenizer_path: Path to tokenizer JSON file
            device: Device to run model on ('cpu' or 'cuda')
            logger: Structured logger instance
        """
        self.logger = logger.bind(component='NameSeqPredictor')
        self.device = device

        # Load tokenizer
        try:
            self.tokenizer = NGramTokenizer.from_file(tokenizer_path)
            self.logger.info('Loaded tokenizer', vocab_size=self.tokenizer.vocab_size())
        except (IOError, AssertionError) as e:
            self.logger.exception('Failed to load tokenizer', tokenizer_path=str(tokenizer_path), exc_info=e)
            raise

        # Load model
        try:
            self.model = NameSeqTransformer.from_trained_model(config_file=config_path,
                                                               model_params_file=model_path,
                                                               tokenizer=self.tokenizer)
            self.model = self.model.to(device)
            self.model.eval()  # Set to evaluation mode
            self.logger.info('Loaded model', model_path=str(model_path), device=device)
        except IOError as e:
            self.logger.exception('Failed to load model', model_path=str(model_path), exc_info=e)
            raise

    def predict_completions(
        self,
        prefix: str,
        num_completions: int,
        top_k: int = 5,
        temperature: float = 1.0
    ) -> list[dict]:
        """Generate completion predictions for a prefix.

        Args:
            prefix: Input prefix string to complete
            num_completions: Number of characters to predict
            top_k: Number of top predictions to return per position
            temperature: Temperature for softmax (lower = sharper)

        Returns:
            List of dicts with keys:
            - 'completion': str - the predicted completion
            - 'confidence': float - average confidence score
            - 'full_sequence': str - prefix + completion
            - 'token_ids': list[int] - token IDs of completion
            - 'probabilities': list[float] - probability for each token
        """
        self.logger.debug('Predicting completions', prefix=prefix, num_completions=num_completions)

        try:
            # Encode prefix
            prefix_tokens = self._encode_prefix(prefix)
            self.logger.debug('Encoded prefix', num_tokens=len(prefix_tokens))

            # Generate predictions iteratively
            completions = []
            current_tokens = prefix_tokens.copy()

            with torch.no_grad():
                for step in range(num_completions):
                    # Get probabilities for next token
                    next_token_probs = self._get_next_token_probabilities(current_tokens)

                    # Get top-k predictions
                    top_probs, top_indices = torch.topk(next_token_probs, k=min(top_k, len(next_token_probs)))
                    top_probs = F.softmax(top_probs / temperature, dim=0)

                    # Sample top predictions as potential completions
                    if step == 0:
                        # First step: create entries for each top prediction
                        for idx, (prob, token_id) in enumerate(zip(top_probs, top_indices)):
                            completions.append({
                                'token_ids': [token_id.item()],
                                'probabilities': [prob.item()],
                                'confidence': prob.item(),
                            })
                    else:
                        # Subsequent steps: extend existing completions
                        new_completions = []
                        for comp in completions:
                            for prob, token_id in zip(top_probs, top_indices):
                                new_comp = {
                                    'token_ids': comp['token_ids'] + [token_id.item()],
                                    'probabilities': comp['probabilities'] + [prob.item()],
                                    'confidence': (comp['confidence'] + prob.item()) / 2,
                                }
                                new_completions.append(new_comp)
                        completions = new_completions

                    # Use greedy best prediction to continue sequence
                    best_token_id = int(top_indices[0].item())
                    current_tokens.append(best_token_id)

            # Decode completions
            results = []
            for comp in completions:
                try:
                    completion_str = self.tokenizer.decode(comp['token_ids'])
                    full_sequence = prefix + completion_str
                    results.append({
                        'completion': completion_str,
                        'full_sequence': full_sequence,
                        'confidence': comp['confidence'],
                        'token_ids': comp['token_ids'],
                        'probabilities': comp['probabilities'],
                    })
                except Exception as e:
                    self.logger.warning('Failed to decode completion', token_ids=comp['token_ids'], exc_info=e)

            # Sort by confidence descending and return top results
            results.sort(key=lambda x: x['confidence'], reverse=True)
            results = results[:top_k]

            self.logger.debug('Generated completions', num_results=len(results))
            return results

        except Exception as e:
            self.logger.exception('Error during prediction', prefix=prefix, exc_info=e)
            raise

    def _encode_prefix(self, prefix: str) -> list[int]:
        """Encode a prefix string into token IDs.

        Args:
            prefix: Prefix string to encode

        Returns:
            List of token IDs

        Raises:
            ValueError: If prefix contains unknown n-grams
        """
        try:
            return self.tokenizer.tokenize(prefix)
        except KeyError as e:
            self.logger.error('Prefix contains unknown tokens', prefix=prefix, exc_info=e)
            raise ValueError(f'Prefix contains unknown tokens: {e}')

    def _get_next_token_probabilities(self, token_ids: list[int]) -> torch.Tensor:
        """Get probability distribution for next token given a sequence.

        Args:
            token_ids: List of token IDs representing input sequence

        Returns:
            Tensor of shape (vocab_size,) with logits for each token
        """
        # Convert to tensor
        token_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)

        # Forward pass through model
        with torch.no_grad():
            # Model expects (batch, seq_len) and produces (batch, seq_len, vocab_size)
            logits = self.model(token_tensor, token_tensor)  # src and tgt are same for inference
            # Get last position's logits
            last_logits = logits[0, -1, :]  # Shape: (vocab_size,)

        return last_logits


class NameSeqCompleterApp(BaseApp[NameSeqCompleterConfig]):
    """Main application orchestrating interactive/batch modes."""

    def __init__(self, argv: Optional[list[str]] = None):
        """Initialize the application.

        Args:
            argv: Command-line arguments (for testing)
        """
        super().__init__(
            arg_template=NameSeqCompleterConfig(),
            description='Complete name sequences using trained transformer models',
            argv=argv
        )
        self.predictor: Optional[NameSeqPredictor] = None

    def run(self):
        """Entry point: dispatches to appropriate mode."""
        try:
            config_dict = asdict(self.config)
            self.logger.info('Starting NameSeqCompleter', **config_dict)

            # Initialize predictor
            self.predictor = NameSeqPredictor(
                model_path=self.config.model_path,
                tokenizer_path=self.config.tokenizer_path,
                config_path=self.config.config_path,
                device=str(select_device()),
                logger=self.logger
            )

            # Dispatch to appropriate mode
            if self.config.mode == 'interactive':
                self._run_interactive_mode()
            elif self.config.mode == 'batch':
                self._run_batch_mode()
            else:
                raise ValueError(f'Unknown mode: {self.config.mode}')

            self.logger.info('Application run completed successfully')

        except Exception as e:
            self.logger.exception('Uncaught error in application', exc_info=e)
            raise

    def _run_interactive_mode(self):
        """Launch interactive REPL."""
        self.logger.info('Entering interactive mode')
        assert self.predictor is not None, 'Predictor must be initialized'

        shell = InteractiveShell(
            predictor=self.predictor,
            config=self.config,
            logger=self.logger
        )
        shell.cmdloop()

    def _run_batch_mode(self):
        """Process batch input file."""
        import json

        self.logger.info('Entering batch mode', input_file=str(self.config.input_file))

        if self.config.input_file is None:
            raise ValueError('input_file required for batch mode')

        input_file = Path(self.config.input_file)
        if not input_file.exists():
            raise FileNotFoundError(f'Input file not found: {input_file}')

        if self.config.output_file is None:
            raise ValueError('output_file required for batch mode')

        output_file = Path(self.config.output_file)

        assert self.predictor is not None, 'Predictor must be initialized'

        # Read input prefixes
        prefixes = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                prefixes = [line.strip() for line in f if line.strip()]
            self.logger.info('Loaded prefixes from file', num_prefixes=len(prefixes))
        except Exception as e:
            self.logger.exception('Error reading input file', exc_info=e)
            raise

        # Generate predictions for each prefix
        results = []
        for i, prefix in enumerate(prefixes, 1):
            self.logger.debug('Processing prefix', prefix_num=i, total=len(prefixes), prefix=prefix)

            try:
                completions = self.predictor.predict_completions(
                    prefix=prefix,
                    num_completions=self.config.max_predictions,
                    top_k=self.config.top_k,
                    temperature=self.config.temperature
                )

                # Format results for output
                result_entry = {
                    'prefix': prefix,
                    'completions': [
                        {
                            'full_sequence': comp['full_sequence'],
                            'confidence': comp['confidence'],
                            'rank': idx + 1
                        }
                        for idx, comp in enumerate(completions)
                    ]
                }
                results.append(result_entry)

            except Exception as e:
                self.logger.error('Error completing prefix', prefix=prefix, exc_info=e)
                # Include error in output
                results.append({
                    'prefix': prefix,
                    'error': str(e),
                    'completions': []
                })

        # Write results to output file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            self.logger.info('Batch processing complete', output_file=str(output_file), num_results=len(results))
        except Exception as e:
            self.logger.exception('Error writing output file', exc_info=e)
            raise


class InteractiveShell(cmd.Cmd):
    """Interactive shell for name sequence completion using cmd.Cmd framework."""

    intro = (
        '\n=== Name Sequence Completer ===\n'
        'Type a prefix to complete it, or use commands below.\n'
        'Type "help" for available commands.\n'
    )
    prompt = 'NSC> '

    def __init__(
        self,
        predictor: NameSeqPredictor,
        config: NameSeqCompleterConfig,
        logger: structlog.BoundLogger
    ):
        """Initialize the interactive shell.

        Args:
            predictor: NameSeqPredictor instance for completions
            config: NameSeqCompleterConfig instance
            logger: Structured logger
        """
        super().__init__()
        self.predictor = predictor
        self.config = config
        self.logger = logger.bind(component='InteractiveShell')

    def default(self, line: str) -> None:
        """Handle default input (prefix completion).

        Any input that doesn't match a do_* command is treated as a prefix
        to complete.

        Args:
            line: User input line
        """
        if not line or line.isspace():
            return

        try:
            prefix = line.strip()
            self.logger.debug('Completing prefix', prefix=prefix)

            results = self.predictor.predict_completions(
                prefix=prefix,
                num_completions=self.config.max_predictions,
                top_k=self.config.top_k,
                temperature=self.config.temperature
            )

            self._display_predictions(prefix, results)

        except Exception as e:
            self.logger.error('Error completing prefix', prefix=line, exc_info=e)
            print(f'Error: {e}')

    def do_complete(self, args: str) -> None:
        """complete <prefix> -- Complete the given prefix string.

        Args:
            args: Prefix string to complete
        """
        if not args or args.isspace():
            print('Usage: complete <prefix>')
            return

        self.default(args)

    def do_config(self, args: str) -> None:
        """config -- Display current configuration settings."""
        print('\nCurrent Configuration:')
        print(f'  model_path: {self.config.model_path}')
        print(f'  tokenizer_path: {self.config.tokenizer_path}')
        print(f'  top_k: {self.config.top_k}')
        print(f'  temperature: {self.config.temperature}')
        print(f'  max_predictions: {self.config.max_predictions}')
        print()

    def do_topk(self, args: str) -> None:
        """topk <N> -- Set the number of top predictions to show (1-vocab_size).

        Args:
            args: Number of top predictions
        """
        if not args or args.isspace():
            print(f'Current top_k: {self.config.top_k}')
            return

        try:
            new_k = int(args.strip())
            if new_k < 1:
                print('Error: top_k must be >= 1')
                return
            self.config.top_k = new_k
            print(f'Updated top_k to {new_k}')
        except ValueError:
            print(f'Error: "{args}" is not a valid integer')

    def help_complete(self) -> None:
        """Help text for complete command."""
        print('complete <prefix> -- Complete the given prefix string.')

    def help_config(self) -> None:
        """Help text for config command."""
        print('config -- Display current configuration settings.')

    def help_topk(self) -> None:
        """Help text for topk command."""
        print('topk <N> -- Set the number of top predictions to show.')

    def _display_predictions(self, prefix: str, results: list[dict]) -> None:
        """Pretty-print predictions.

        Args:
            prefix: Original prefix
            results: List of prediction results
        """
        print(f'\nCompletions for "{prefix}":')
        if not results:
            print('  (no predictions)')
            return

        for i, result in enumerate(results, 1):
            confidence = result['confidence']
            completion = result['completion']
            full = result['full_sequence']
            print(f'  {i}. {full:30s} [{confidence:.3f}]')
        print()


if __name__ == '__main__':
    NameSeqCompleterApp().run()
