# Copilot Instructions for torch_playground

This document outlines the architecture, patterns, and workflows for the torch_playground codebase, which implements a progressive series of PyTorch learning exercises.

## Project Overview

This codebase implements a series of increasingly complex PyTorch exercises, focusing on:
- Progressive learning through practical examples
- Clear, well-structured implementation patterns
- Quality engineering practices (testing, logging, configuration)

For Python code style and syntax conventions, see `.github/instructions/copilot-instructions.md.instructions.md`.

## Project Structure

- **Demo Programs (`src/torch_playground/d[0-9]+.py`)**
  - Numbered exercises with increasing complexity
  - Each demo introduces new PyTorch concepts
  - Uses synthetic data for learning focus
  - Example progression:
    1. `d00_linear_projection.py`: Basic linear operations
    2. `d01_linear_trainable.py`: Adding training loops
    3. `d02_linear_multilayer.py`: Multi-layer networks
    4. `d03_basic_transformer.py`: Transformer architecture

- **Core Infrastructure**
  - `util.py`: Application framework, logging, training loops
  - `tokenizer.py`: Text processing utilities

- **Tests & Quality**
  - Unit tests: `tests/test_*.py`
  - Integration tests: `tests/test_integration_*.py`
  - Coverage requirements in pyproject.toml

## Architecture & Design

1. **Application Framework**
   The codebase uses a hierarchical application framework:
   ```
   BaseApp                          # Core infrastructure
     └── TrainableModelApp         # Adds training capabilities
           └── Demo Apps           # Individual exercises
   ```

   Key features:
   - Standardized CLI argument handling
   - Structured logging with context
   - Output management (checkpoints, logs, tensorboard)
   - Configurable training loops

2. **Model Architecture**
   Models follow a consistent pattern:
   - Configuration class defines model parameters
   - Model class implements the architecture
   - Factory method handles instantiation
   - Standard initialization and device handling

3. **Training Infrastructure**
   Standardized training workflow:
   ```
   App.run()
     ├── Model initialization from config
     ├── Data loading and preprocessing
     ├── Training loop with metrics
     └── Model checkpointing and analysis
   ```

   Built-in features:
   - Progress tracking with tqdm
   - TensorBoard metric logging
   - Checkpoint management
   - Structured error handling

## Key Development Workflows

1. **Dependency Management**
  - This is a `uv`-based project.
  - Install dependencies with `uv add [dependency name]`.
  - Execute code with `uv run`
    - Example: `uv run pytest`
    - Example: `uv run python src/torch_playground/app_demo.py --help`

1. **Setup & Environment**
   ```bash
   # Install dependencies
   uv sync
   ```

   Note: Project requires Python 3.12+ and PyTorch 2.2 (Intel Mac compatibility)

2. **Development Process**
   New features follow this workflow:
   1. Create new demo in numbered sequence
   2. Add matching test files
   3. Implement using standard patterns
   4. Validate with test suite
   5. Document key learnings

3. **Data Pipeline Patterns**
   ```python
   # Standard data loading pattern
   data = FileDataset(data_path)\
       .with_transform(preprocess)\
       .with_transform(tokenize)
   loader = DataLoader(data, batch_size=config.batch_size)
   ```

4. **Output Management**
   Each run creates a timestamped directory containing:
   ```
   <timestamp>/
     ├── config.json           # Run configuration
     ├── model_summary.txt     # Architecture details
     ├── logs/                 # Structured logs
     ├── checkpoints/         # Model snapshots
     └── tensorboard/        # Training metrics
   ```

## Common Development Tasks

1. **Creating a New Demo**
   ```bash
   # 1. Create new demo file d<XX>_<name>.py
   # 2. Create matching test files:
   touch tests/test_d<XX>_<name>.py
   touch tests/test_integration_d<XX>.py
   ```

2. **Running Models**
   ```bash
   # Basic run with defaults
   python -m src.torch_playground.d<XX>_<name>

   # Common options
   python -m src.torch_playground.d<XX>_<name> \
     --loglevel DEBUG \
     --output_dir /path/to/output \
     --randseed 42
   ```

3. **Debugging**
   - Logs in `<output_dir>/<timestamp>/logs/`
   - TensorBoard metrics in `<output_dir>/tensorboard/<timestamp>/`
   - Model summaries in `<output_dir>/<timestamp>/model_summary.txt`