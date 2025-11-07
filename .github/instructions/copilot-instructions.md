---
applyTo: '**.py'
---
# Python Code Style & Convention Instructions

Follow these conventions for all Python files in the torch_playground codebase:

## Code Style

1. **String Quotes**
   - Use single quotes for string literals unless double quotes are necessary
   - Example: Use `'example'` instead of `"example"`
   - Exception: Use double quotes for strings containing single quotes: `"It's a test"`

2. **Type Hints**
   - Always include type hints for function parameters and return values
   - Use Python 3.12+ syntax for type hints (e.g., `list[str]` not `List[str]`)
   - Example:
   ```python
   def process_data(input_tensor: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
   ```

3. **Documentation**
   - All classes and public methods must have docstrings
   - Use """triple quotes""" for docstrings (exception to single quote rule)
   - Include Args/Returns sections in function docstrings
   - TODOs must be formatted as `# TODO(username): description`
   - Example:
   ```python
   def train_model(self, data: DataLoader, optimizer: torch.optim.Optimizer) -> None:
       """Train the model using the provided data and optimizer.

       Args:
           data: DataLoader providing (input, target) batches
           optimizer: The configured optimizer instance
       """
   ```

4. **Project Configuration**
   - Put Python toolchain configuration in pyproject.toml when possible
   - This includes pytest, coverage, and flake8 settings

5. **Class Structure**
   - Configuration classes use dataclass decorator with explicit field metadata
   - Model classes include `from_config` factory method
   - Example:
   ```python
   @dataclass
   class ModelConfig(BaseConfiguration):
       hidden_size: int = field(
           default=128,
           metadata=BaseConfiguration._meta(help='Size of hidden layers')
       )
   ```

6. **File layout style**
  - Put two blank lines between top-level classes or standalone methods in a module.
  - Put two blank lines following imports block.
  - Put two blank lines following any block of module-level constants or types.
  - Put one blank line between methods within a class.
  - Alphabetize imports by module name.
  - End the file with a newline.

## Error Handling & Logging

1. **Logging**
   - Use `self.logger` from BaseApp for consistent logging
   - Include relevant context in log messages
   - Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
   - Example:
   ```python
   self.logger.debug('Processing batch', batch_id=batch_id, batch_size=len(batch))
   ```

2. **Error Handling**
   - Validate configuration parameters in `__init__`
   - Use descriptive error messages
   - Log exceptions before re-raising
   - Example:
   ```python
   if config.n_heads > config.d_model:
       raise ValueError(f'Number of heads ({config.n_heads}) cannot exceed model dimension ({config.d_model})')
   ```
