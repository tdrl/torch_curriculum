"""Shared utility functions."""

import keyring  # Accessing API keys securely from Apple Keychain or Windows Credential Store.
import structlog
from structlog.processors import TimeStamper, StackInfoRenderer, UnicodeDecoder
from structlog.dev import ConsoleRenderer, RichTracebackFormatter, RED, GREEN, BLUE, MAGENTA, YELLOW, CYAN, RED_BACK, BRIGHT
from structlog.stdlib import add_log_level, PositionalArgumentsFormatter
import logging
import logging.config
from dataclasses import dataclass, fields, field
import argparse
from typing import Optional
import os
import sys
from pathlib import Path
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import pprint
import datetime

__all__ = [
    'setup_logging',
    'fetch_api_keys',
    'BaseArguments',
    'parse_cmd_line_args',
]

def get_default_working_dir() -> Path:
    """Get the default working directory based on the current script name."""
    return Path('/tmp/') / os.getenv('USER', 'unknown_user') / Path(sys.argv[0]).stem


@dataclass
class BaseArguments:
    """Root class for command line arguments.

    Command line flags are defined as dataclass fields, where the type, default value, and help text are specified
    and passed to the argparse parser in parse_cmd_line_args.

    This class serves as a base for all command line argument classes.
    It includes common fields such as log level and log directory.
    Implementations should subclass this class and define additional fields as needed.
    IMPORTANT: All subclasses must also be declared as dataclasses for their fields to be
    recognized by parse_cmd_line_args().
    """
    @staticmethod
    def _meta(help: Optional[str] = None, required: bool = False):
        """Helper method to define metadata for dataclass fields."""
        if help is None:
            help_str = ''
        else:
            help_str = help + ' '
        return {
            'help': help_str + '(default: %(default)s)',
            'required': required,
        }

    loglevel: str = field(default='INFO',
                          metadata=_meta(help='Logging level for stdout logs (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)'))
    output_dir: Path = field(default=get_default_working_dir(),
                             metadata=_meta(help='Directory to save the data, checkpoints, trained model, etc.'))
    logdir: Path = field(default=get_default_working_dir() / 'logs',
                         metadata=_meta(help='Directory where log files will be stored.'))
    randseed: int = field(default=9_192_631_770,  # Frequency of ground state hyperfine transition of cesium-133 in Hz.
                          metadata=_meta(help='Random seed for reproducibility.'))


def parse_cmd_line_args[T: BaseArguments](arg_template: T, description: Optional[str], argv: Optional[list[str]]) -> T:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=description)
    for field in fields(arg_template):
        parser.add_argument(f'--{field.name}',
                            type=field.type,
                            default=field.default,
                            help=field.metadata.get('help', f'{field.default}'),
                            required=field.metadata.get('required', False))
    args = parser.parse_args(argv, namespace=arg_template)
    return args


def setup_logging(loglevel: str = 'INFO',
                  logdir: Path = get_default_working_dir() / 'logs') -> structlog.BoundLogger:
    """Set up logging configuration.

    Sets a local logger and configures the logging format.

    Args:
        args (BaseArguments): The command line arguments containing log level and log directory.
          This function consumes the fields:
            - loglevel: The logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
            - logdir: The directory where log files will be stored.

    Returns:
        structlog.BoundLogger: A logger instance configured with the specified settings.
    """
    # Suppress asyncio 'KqueueSelector' messages.
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    # Ensure logdir exists
    logdir.mkdir(parents=True, exist_ok=True)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt='iso'),
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'console_colored': {
                '()': structlog.stdlib.ProcessorFormatter,
                'processor': structlog.dev.ConsoleRenderer(colors=True),
            },
            'json': {
                '()': structlog.stdlib.ProcessorFormatter,
                'processor': structlog.processors.JSONRenderer(),
            },
        },
        'handlers': {
            'console': {
                'level': loglevel,
                'class': 'logging.StreamHandler',
                'formatter': 'console_colored',
            },
            'file_debug': {
                'level': 'DEBUG',
                'class': 'logging.handlers.WatchedFileHandler',
                'filename': str(logdir / 'debug.json.log'),
                'formatter': 'json',
            },
            'file_error': {
                'level': 'ERROR',
                'class': 'logging.handlers.WatchedFileHandler',
                'filename': str(logdir / 'error.json.log'),
                'formatter': 'json',
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file_debug', 'file_error'],
                'level': 'DEBUG',
                'propagate': True,
            },
        },
    })
    return structlog.get_logger(__name__)


def fetch_api_keys() -> dict[str, str]:
    """Fetch the API key for Hugging Face from the system's keyring.

    Currently, it retrieves the Hugging Face access token.

    Returns:
        dict[str, str]: A dictionary mapping service names to API keys.
            If no keys are found, returns an empty dictionary. Currently supported keys:
                - 'huggingface': The Hugging Face API access token.
    """
    api_key = keyring.get_password('net.illation.heather/huggingface/exploration',
                                   'studentbane')
    return {'huggingface': api_key} if api_key is not None else {}


class App[T: BaseArguments]:
    """Base class for applications.

    This class provides a common interface for applications, including methods for running the application
    and setting up logging.
    """
    def __init__(self, arg_template: T, description: Optional[str], argv: Optional[list[str]] = None):
        self.config = parse_cmd_line_args(arg_template=arg_template, description=description, argv=argv)
        self.logger = setup_logging(self.config.loglevel, self.config.logdir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.debug('Assigned device', device=self.device)
        torch.manual_seed(self.config.randseed)
        self.logger.debug('Set random seed', randseed=self.config.randseed)
        self.run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        self.tb_writer = SummaryWriter(log_dir=self.config.output_dir / 'tensorboard' / self.run_timestamp)

    def run(self):
        """Run the application."""
        # TODO(heather): Wrap this in a decorator that catches exceptions and logs them.
        raise NotImplementedError("Subclasses must implement this method.")


def save_tensor(tensor: torch.Tensor, path: Path):
    """Save a tensor to both pt and txt files."""
    # TODO(heather): Add tests for this function.
    # TODO(heather): Move to safetenors representation.
    torch.save(tensor, path.with_suffix('.pt'))
    with open(path.with_suffix('.txt'), 'w') as f:
        f.write(pprint.pformat(tensor.tolist(), indent=2, width=80))
