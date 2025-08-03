"""Shared utility functions."""

import keyring  # Accessing API keys securely from Apple Keychain or Windows Credential Store.
import structlog
from structlog.processors import TimeStamper, StackInfoRenderer, UnicodeDecoder
from structlog.dev import ConsoleRenderer, RichTracebackFormatter, RED, GREEN, BLUE, MAGENTA, YELLOW, CYAN, RED_BACK, BRIGHT
from structlog.stdlib import add_log_level, PositionalArgumentsFormatter

from dataclasses import dataclass, fields, field
import argparse
from typing import Optional
import os
from pathlib import Path

__all__ = [
    'setup_logging',
    'fetch_api_keys',
    'BaseArguments',
    'parse_cmd_line_args',
]


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
    loglevel: str = field(default='INFO',
                          metadata={
                              'help': 'Logging level for stdout logs (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: %(default)s)'
                          })
    logdir: Path = field(default=Path('/tmp/') / os.getenv('USER', 'unknown_user'),
                         metadata={
                              'help': 'Directory where log files will be stored. (default: %(default)s).'
                         })


def parse_cmd_line_args[T: BaseArguments](arg_template: T, argv: Optional[list[str]]) -> T:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    for field in fields(arg_template):
        parser.add_argument(f'--{field.name}',
                            type=field.type,
                            default=field.default,
                            help=field.metadata.get('help', f'{field.default}'),
                            required=field.metadata.get('required', False))
    args = parser.parse_args(argv, namespace=arg_template)
    return args


def setup_logging(args: BaseArguments) -> structlog.BoundLogger:
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
    # Ensure logdir exists
    logdir = args.logdir
    logdir.mkdir(parents=True, exist_ok=True)

    # Helper: ISO-8601 time
    timestamper = TimeStamper(fmt='iso', utc=True)

    # # File sinks
    # debug_file = open(logdir / 'debug.log', 'a', encoding='utf-8')
    # warning_file = open(logdir / 'warning.log', 'a', encoding='utf-8')

    # def file_sink_factory(fileobj, min_level):
    #     def sink(event_dict):
    #         if event_dict['level'] >= min_level:
    #             # Format: ISO-8601|LEVEL|LOGGER> message
    #             ts = event_dict.get('timestamp', '')
    #             lvl = event_dict.get('level_name', '')
    #             name = event_dict.get('logger', '')
    #             msg = event_dict.get('event', '')
    #             fileobj.write(f'{ts}|{lvl}|{name}> {msg}\n')
    #             fileobj.flush()
    #         return event_dict
    #     return sink

    # # Stdout sink
    # def stdout_sink(event_dict):
    #     if event_dict['level'] >= level_stdout:
    #         ts = event_dict.get('timestamp', '')
    #         lvl = event_dict.get('level_name', '')
    #         name = event_dict.get('logger', '')
    #         msg = event_dict.get('event', '')
    #         print(f'{ts}|{lvl}|{name}> {msg}', file=sys.stdout)
    #     return event_dict

    # Compose processors
    processors = [
        timestamper,
        add_log_level,
        PositionalArgumentsFormatter(),
        StackInfoRenderer(),
        UnicodeDecoder(),
        ConsoleRenderer(colors=True,
                         level_styles={
            'debug': BLUE,
            'info': GREEN,
            'warning': YELLOW,
            'error': RED,
            'critical': RED_BACK,
        }),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
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


class App:
    """Base class for applications.

    This class provides a common interface for applications, including methods for running the application
    and setting up logging.
    """
    def __init__(self, arg_template: BaseArguments, argv: Optional[list[str]] = None):
        self.args = parse_cmd_line_args(arg_template=arg_template, argv=argv)
        self.logger = setup_logging(self.args)

    def run(self):
        """Run the application."""
        raise NotImplementedError("Subclasses must implement this method.")