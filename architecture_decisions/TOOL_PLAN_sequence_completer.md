# Plan: Name Sequence Completer Tool

## Executive Summary

This document outlines the implementation plan for a new tool (`name_seq_completer`) that loads trained models from `d04_name_seq_learner.py` and uses them to complete prefix strings through sequence prediction. The tool will support both interactive shell mode and batch processing mode, following the established patterns in the torch_playground codebase.

---

## Requirements Analysis

### Functional Requirements

1. **Model Loading**
   - Load serialized models saved by `d04_name_seq_learner.py` (format: PyTorch `.pt` files)
   - Load associated tokenizers used during training
   - Support for loading from specified checkpoint directory

2. **Interactive Shell Mode**
   - Provide a read-eval-print loop (REPL) for user interaction using Python's `cmd.Cmd` framework
   - Accept user-provided prefix strings
   - Display predictions with confidence scores
   - Automatic command history and line editing (via `readline` module, integrated with `cmd.Cmd`)
   - Built-in help system and exit handling from `cmd.Cmd`
   - Custom commands: `complete`, `config`, `topk`
   - Configurable number of top predictions to show

3. **Batch Mode**
   - Read input prefixes from file or stdin
   - Process prefixes non-interactively
   - Output predictions in structured format (e.g., JSON)
   - Configurable output format

4. **Prediction Engine**
   - Use the model to generate token predictions for each position
   - Decode token IDs back to strings using tokenizer
   - Track and display confidence/probability scores
   - Handle variable-length sequences

### Non-Functional Requirements

- Follow the existing `torch_playground` architecture patterns
- Integrate with existing logging infrastructure
- Support same CLI argument conventions as other apps
- Comprehensive test coverage
- Clear documentation and error handling

---

## Architecture & Design

### 1. Core Components

#### 1.1 `NameSeqCompleterConfig` (Configuration)

**Location:** `src/torch_playground/name_seq_completer.py`

Extends `BaseConfiguration` with tool-specific parameters:

```python
@dataclass
class NameSeqCompleterConfig(BaseConfiguration):
    # Model/Tokenizer I/O
    model_path: Path              # Path to trained model (.pt file)
    tokenizer_path: Path          # Path to tokenizer file (.json)

    # Prediction parameters
    max_predictions: int          # Number of completions to generate
    top_k: int                    # Top-K predictions to show (default: 5)
    temperature: float            # Softmax temperature for probability scaling

    # Mode selection
    mode: str                     # 'interactive' or 'batch'
    input_file: Optional[Path]    # For batch mode: input prefixes file
    output_file: Optional[Path]   # For batch mode: output predictions file
    output_format: str            # 'json' or 'text' for batch mode
```

#### 1.2 `NameSeqPredictor` (Prediction Engine)

**Location:** `src/torch_playground/name_seq_completer.py`

Core class handling model-based predictions:

```python
class NameSeqPredictor:
    """Wrapper around NameSeqTransformer for completion predictions.

    Responsibilities:
    - Load model and tokenizer
    - Handle prefix encoding
    - Generate predictions using the model
    - Decode token IDs back to strings
    - Score predictions (probabilities/confidence)
    """

    def __init__(self, model_path: Path, tokenizer_path: Path, device: str)

    def predict_completions(self, prefix: str,
                           num_completions: int,
                           top_k: int = 5,
                           temperature: float = 1.0) -> list[dict]
        """Generate completion predictions for a prefix.

        Returns list of dicts: [
            {'completion': str, 'confidence': float, 'full_sequence': str},
            ...
        ]
        """

    def _encode_prefix(self, prefix: str) -> torch.Tensor
        """Tokenize prefix string into token tensor."""

    def _decode_tokens(self, tokens: torch.Tensor) -> str
        """Convert token tensor back to string."""

    def _get_next_token_probabilities(self, src: torch.Tensor) -> torch.Tensor
        """Run model forward pass and get probability distribution for next token."""
```

#### 1.5 Tokenizer Extensions (`tokenizer.py`)

**Location:** `src/torch_playground/tokenizer.py` (extending existing `NGramTokenizer`)

Add decoding method to existing `NGramTokenizer` class:

```python
def decode(self, token_ids: list[int] | torch.Tensor) -> str:
    """Decode a sequence of token IDs back to a string.

    Args:
        token_ids: List of token IDs or torch.Tensor of shape (seq_len,)
    Returns:
        str: Decoded string by concatenating token strings
    Raises:
        ValueError: If a token ID is not in the vocabulary
    """

def get_token_string(self, token_id: int) -> str | None:
    """Get the string representation of a token ID.

    Args:
        token_id (int): Token ID to look up
    Returns:
        str | None: Token string, or None if ID not in vocabulary
    """

def has_token_id(self, token_id: int) -> bool:
    """Check if a token ID exists in the vocabulary.

    Args:
        token_id (int): Token ID to check
    Returns:
        bool: True if token ID is valid
    """
```

**Design Notes:**
- Tokenizer is responsible **only** for converting integer IDs to strings
- Probability handling and selecting token IDs from probability distributions remains in `name_seq_completer.py`
- Decoding is inverse of tokenization: takes token IDs, returns strings
- Supports both list and torch.Tensor inputs for convenience
- Provides helper methods for token lookup and validation

#### 1.3 `NameSeqCompleterApp` (Application)

**Location:** `src/torch_playground/name_seq_completer.py`

Main application class, extends `BaseApp`:

```python
class NameSeqCompleterApp(BaseApp[NameSeqCompleterConfig]):
    """Main application orchestrating interactive/batch modes.

    Type Parameters:
        T = NameSeqCompleterConfig
    """

    def __init__(self, argv: Optional[list[str]] = None)

    def run(self)
        """Entry point: dispatches to appropriate mode."""

    def _run_interactive_mode(self)
        """Launch interactive REPL."""

    def _run_batch_mode(self)
        """Process batch input file."""
```

#### 1.4 `InteractiveShell` (User Interaction)

**Location:** `src/torch_playground/name_seq_completer.py`

REPL interface built on `cmd.Cmd`:

```python
class InteractiveShell(cmd.Cmd):
    """Read-eval-print loop for interactive completion using cmd.Cmd framework.

    Inherits from cmd.Cmd to provide:
    - Command history and line editing (via readline)
    - Automatic help system
    - Built-in quit/exit handling
    - Clean command dispatch mechanism

    Custom Commands:
    - complete <prefix>    Complete the given prefix (default action)
    - config              Show current configuration
    - topk <N>            Change top_k value for current session
    - quit / exit / EOF   Exit the shell
    """

    def __init__(self, predictor: NameSeqPredictor, config: NameSeqCompleterConfig, logger: structlog.BoundLogger)

    def default(self, line: str) -> None
        """Called when user input doesn't match any do_* command.
        Treats input as a prefix for completion."""

    def do_complete(self, args: str) -> None
        """complete <prefix> -- Complete the given prefix string."""

    def do_config(self, args: str) -> None
        """config -- Display current configuration settings."""

    def do_topk(self, args: str) -> None
        """topk <N> -- Set the number of top predictions to show (1-vocab_size)."""

    def help_complete(self) -> None
        """Help text for complete command."""

    def help_config(self) -> None
        """Help text for config command."""

    def help_topk(self) -> None
        """Help text for topk command."""

    def _display_predictions(self, results: list[dict]) -> None
        """Pretty-print predictions using rich library."""

    def _validate_input(self, prefix: str) -> bool
        """Validate prefix string (check for unknown characters, length, etc.)."""
```

### 2. Module Organization

```
src/torch_playground/
├── name_seq_completer.py       # Main tool (Config, Predictor, App, Shell)
└── (existing files unchanged)

tests/
├── test_name_seq_completer.py  # Unit tests
└── test_integration_completer.py  # Integration tests
```

---

```

## Implementation Phases

### Phase 0: Tokenizer Extension

**Deliverables:** Token decoding methods in `NGramTokenizer`

1. Add `decode()` method to `NGramTokenizer` class
   - Accepts list or torch.Tensor of token IDs
   - Returns concatenated string from token strings
   - Handles unknown token IDs gracefully
2. Add `get_token_string()` helper method
   - Look up string representation by token ID
   - Returns None if ID not in vocabulary
3. Add `has_token_id()` validation method
   - Check if token ID exists in vocabulary
4. Write unit tests for decoding methods
   - Test single token, sequence, tensor input
   - Test error handling for invalid IDs
   - Test round-trip: string → encode → decode → string

**Key Design Principle:** Tokenizer handles ID↔string conversion only; probability handling stays in `NameSeqPredictor`

### Phase 1: Core Prediction Engine

**Deliverables:** `NameSeqPredictor`, basic model loading

1. Create `NameSeqCompleterConfig` dataclass
2. Implement `NameSeqPredictor` class:
   - Model and tokenizer loading
   - Prefix encoding using `tokenizer.tokenize()`
   - Token decoding using `tokenizer.decode()`
   - Forward pass for next-token prediction
   - Probability computation and sorting
3. Write unit tests for prediction correctness

**Key Patterns from Codebase:**

- Model loading: See `d04_name_seq_learner.py` lines ~85-95
- Tokenizer: Use extended `NGramTokenizer` with decoding capabilities
- Device handling: Follow `TrainableModelApp.__init__()` pattern for device selection

### Phase 2: Batch Processing Mode

**Deliverables:** Batch mode operation via CLI

1. Implement `_run_batch_mode()` in `NameSeqCompleterApp`
2. Input/output file handling
3. Format support:
   - Input: One prefix per line (text file)
   - Output: JSON format with predictions
4. Write integration tests

**Output Format (JSON):**

```json
[
  {
    "prefix": "John",
    "completions": [
      {
        "full_sequence": "Johnas",
        "confidence": 0.92,
        "top_k_rank": 1
      },
      ...
    ]
  }
]
```

### Phase 3: Interactive Shell Mode

**Deliverables:** Interactive REPL using `cmd.Cmd` framework

1. Implement `InteractiveShell` class extending `cmd.Cmd`
2. Override `default()` method to handle prefix completion as default action
3. Implement custom commands: `complete`, `config`, `topk`
4. Help text via `help_*` methods (automatic in `cmd.Cmd`)
5. Readline integration for history and editing (automatic in `cmd.Cmd`)
6. Pretty-printing with `rich` library
7. Write unit tests for shell commands and input handling

**Shell Behavior:**

```text
(NameSeqCompleter) help

Documented commands (type help <topic>):
========================================
complete  config  help  quit  topk

(NameSeqCompleter) help complete
complete <prefix> -- Complete the given prefix string.

(NameSeqCompleter) John
Completions for "John":
  1. Johnas      [0.92]
  2. Johnson     [0.88]
  ...

(NameSeqCompleter) config
Current Configuration:
  top_k: 5
  temperature: 1.0
  max_predictions: 10

(NameSeqCompleter) topk 3
Updated top_k to 3

(NameSeqCompleter) quit
```

### Phase 4: Integration & Polish

**Deliverables:** Fully functional tool

1. Complete `NameSeqCompleterApp.run()` dispatcher
2. Error handling and validation:
   - Missing model/tokenizer files
   - Invalid prefix formats
   - Device availability
3. Documentation and docstrings
4. Configuration validation

---

## Design Decisions & Rationale

### 1. Extension of `BaseApp` vs. `TrainableModelApp`

**Decision:** Extend `BaseApp` directly (not `TrainableModelApp`)

**Rationale:**

- This is an inference-only tool, not a training app
- No optimizer, loss function, training loop needed
- `TrainableModelApp` overhead is unnecessary
- Cleaner separation of concerns

### 2. Using `cmd.Cmd` for Interactive Shell

**Decision:** Inherit `InteractiveShell` from Python's standard library `cmd.Cmd`

**Rationale:**

- **Battle-tested**: `cmd.Cmd` is part of Python's standard library, stable and well-documented
- **Automatic readline integration**: Inheriting from `cmd.Cmd` automatically provides command history, line editing, and tab completion
- **Built-in help system**: Help text generated automatically from docstrings via `help_*` methods
- **Command dispatch**: Automatic routing from user input to `do_*` methods reduces manual parsing
- **Familiar to users**: Anyone familiar with Python REPL or interactive shells knows this pattern
- **Lower maintenance**: Less custom code = fewer bugs and easier testing
- **Extensible**: Easy to add new commands by just adding new `do_*` methods

Alternative considered: Custom REPL using manual readline/input loops
- Would require more boilerplate code for history, editing, help
- `cmd.Cmd` is the idiomatic Python approach for CLIs

### 3. Single File for Tool Implementation

**Decision:** All components in single `name_seq_completer.py`

**Rationale:**

- Tool is relatively self-contained
- Facilitates unit testing (everything in one place)
- Follows pattern of demo files (each demo is self-contained)
- Can be modularized later if necessary

### 4. Tokenizer: String↔ID Conversion Only

**Decision:** Tokenizer extension adds `decode()` method; probability handling stays in `NameSeqPredictor`

**Rationale:**

- **Single Responsibility**: Tokenizer handles bidirectional ID↔string mapping, nothing else
- **Reusability**: Decoding is useful beyond this tool (logging, debugging, other inference tools)
- **Testability**: Can test tokenizer independently from model inference
- **Separation of Concerns**:
  - Tokenizer: `string ↔ token_ids` conversion
  - NameSeqPredictor: `token_ids + probabilities → ranked_predictions`
- **Maintainability**: Changes to probability handling don't affect tokenizer; vice versa

### 5. Prediction Confidence Scoring

**Decision:** Use Softmax probabilities as confidence scores

**Rationale:**

- Standard approach for classification/next-token prediction
- Allows temperature-based control of sharpness
- Interpretable as probability distribution
- Can be aggregated across sequence positions if needed

### 6. Batch vs. Interactive Dispatch

**Decision:** Command-line flag `--mode` selects behavior

**Rationale:**

- Follows existing pattern in torch_playground (e.g., config flags)
- Clear, explicit, easy to test
- Can be extended to support YAML config files later

---

## Testing Strategy

### Unit Tests for Tokenizer Extension (`tests/test_tokenizer.py`)

1. **Decoding Tests** (new methods in `NGramTokenizer`)
   ```python
   test_decode_single_token()             # Single token ID → string
   test_decode_sequence()                 # List of token IDs → concatenated string
   test_decode_tensor()                   # torch.Tensor input handling
   test_decode_invalid_token_id()         # Error handling for unknown IDs
   test_round_trip()                      # string → tokenize → decode → string
   ```

2. **Token Lookup Tests** (new methods in `NGramTokenizer`)
   ```python
   test_get_token_string_valid()          # Valid token ID lookup
   test_get_token_string_invalid()        # Returns None for invalid IDs
   test_has_token_id_valid()              # True for valid token IDs
   test_has_token_id_invalid()            # False for invalid token IDs
   ```

### Unit Tests (`tests/test_name_seq_completer.py`)

1. **Prediction Engine Tests**
   ```python
   test_predictor_initialization()        # Load model/tokenizer
   test_predict_completions_format()      # Check output format
   test_encode_decode_roundtrip()         # Tokenizer consistency
   test_temperature_scaling()             # Probability distribution changes
   test_edge_cases()                      # Empty prefix, very long prefix, unknown chars
   ```

2. **Configuration Tests**
   ```python
   test_config_parsing_interactive()
   test_config_parsing_batch()
   test_config_validation()               # Required fields, type checking
   test_device_selection()                # CPU/CUDA availability
   ```

3. **Shell Logic Tests**
   ```python
   test_command_parsing()                 # Parse special commands
   test_help_display()
   test_config_display()
   test_topk_parameter_update()
   ```

### Integration Tests (`tests/test_integration_completer.py`)

1. **End-to-End Interactive**
   - Launch app in interactive mode
   - Send prefixes via stdin
   - Verify prediction format and quality

2. **End-to-End Batch**
   - Create test input file
   - Run batch mode
   - Validate output JSON structure

3. **Model Loading**
   - Use trained model from d04_name_seq_learner
   - Verify predictions are reasonable

---

## Dependencies & Setup

### Existing Dependencies (Already in `pyproject.toml`)

- torch (2.2)
- structlog (logging)
- pytest (testing)
- rich (pretty printing)

### Standard Library Modules Used

- `cmd`: Interactive shell framework (no additional install needed)
- `readline`: Command history and line editing (automatically integrated with `cmd.Cmd`)

### New Dependencies to Add (If Needed)

None anticipated! Both `cmd` and `readline` are in Python stdlib.

### Environment Setup
```bash
# Build tokenizer (if not already done)
uv run python -m torch_playground.build_tokenizer

# Train d04 model (if not already done)
uv run python -m torch_playground.d04_name_seq_learner --names_file <path> --tokenizer_file <path> --epochs 50

# Run tool (once implemented)
uv run python -m torch_playground.name_seq_completer --mode interactive --model_path <path>
```

---

## Error Handling & Edge Cases

### Error Scenarios to Handle

1. **Model/Tokenizer Not Found**
   - Clear error message with expected path
   - Suggest user provide correct paths

2. **Invalid Prefix**
   - Character not in tokenizer vocabulary
   - Empty string
   - Very long sequence (> max sequence length)

3. **Model Loading Failure**
   - Incompatible model architecture
   - Corrupted checkpoint file
   - Device mismatch (e.g., CUDA model on CPU-only system)

4. **Batch Mode**
   - Input file not found
   - Output directory not writable
   - Malformed JSON output on previous runs

### Edge Cases

1. **Single-character prefix** → Should still generate completions
2. **Prefix already at max length** → Return empty completion set with warning
3. **Repeated completions** → Handle gracefully (different confidences)
4. **Batch mode with empty input** → Warn and process gracefully

---

## Future Extensions

These are out of scope for Phase 1 but worth noting:

1. **Beam Search Decoding** - Instead of greedy argmax
2. **Constrained Decoding** - Only allow valid names from training set
3. **Multiple Model Ensemble** - Combine predictions from multiple checkpoints
4. **REST API** - Expose tool via FastAPI or Flask
5. **Streaming Mode** - Support reading from pipe/stream
6. **Configuration Files** - Support `.yaml` config for complex setups

---

## Success Criteria

- ✅ Tool loads trained d04 model successfully
- ✅ Predictions are ranked by confidence
- ✅ Interactive mode supports command history and editing
- ✅ Batch mode processes file input and produces JSON output
- ✅ Test coverage ≥ 85% (excluding main entry point)
- ✅ All errors handled gracefully with informative messages
- ✅ Code follows torch_playground patterns and style
- ✅ Integration tests pass with real trained model

---

## Timeline Estimate

| Phase | Tasks | Estimated Time |
|-------|-------|-----------------|
| 0     | Tokenizer extension (decode, lookup) | 1-1.5 hours |
| 1     | Core predictor + tests | 2-3 hours |
| 2     | Batch mode + integration tests | 1-2 hours |
| 3     | Interactive shell + REPL tests | 2-3 hours |
| 4     | Integration, polish, documentation | 1-2 hours |
| **Total** | | **7-11.5 hours** |

---

## References

### Relevant Codebase Files

- `tokenizer.py` - `NGramTokenizer` class (extending with decode methods)
- `d04_name_seq_learner.py` - Model architecture, training
- `util.py` - BaseApp, BaseConfiguration patterns
- `test_tokenizer.py` - Example tokenizer test patterns

### External Resources
- PyTorch Inference: https://pytorch.org/docs/stable/notes/inference_optimization.html
- Python Readline: https://docs.python.org/3/library/readline.html
- Rich Library: https://rich.readthedocs.io/
