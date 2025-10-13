"""Shared utility functions."""

import keyring  # Accessing API keys securely from Apple Keychain or Windows Credential Store.
import structlog
from structlog.processors import TimeStamper, StackInfoRenderer, UnicodeDecoder
from structlog.dev import ConsoleRenderer, RichTracebackFormatter, RED, GREEN, BLUE, MAGENTA, YELLOW, CYAN, RED_BACK, BRIGHT
from structlog.stdlib import add_log_level, PositionalArgumentsFormatter
import logging
import logging.config
from dataclasses import dataclass, fields, field, asdict
import argparse
from typing import Optional
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import pprint
import datetime
import tqdm
import json
from abc import ABC, abstractmethod

__all__ = [
    'setup_logging',
    'fetch_api_keys',
    'BaseConfiguration',
    'parse_cmd_line_args',
    'TrainableModelApp',
    'save_tensor',
]


def get_default_working_dir() -> Path:
    """Get the default working directory based on the current script name."""
    return (Path('/tmp/') /
            os.getenv('USER', 'unknown_user') /
            Path(sys.argv[0]).stem)


@dataclass
class BaseConfiguration:
    """Root class for configurations (e.g., from command line arguments).

    Config parameters are defined as dataclass fields, where the type, default value,
    and help text (for CLI --help) are specified. From there, during initialization,
    they're passed to the argparse parser in parse_cmd_line_args.

    This class serves as a base for all config variable classes.
    It includes common fields such as log level and log directory.
    Implementations should subclass this class and define additional fields as needed.
    IMPORTANT: All subclasses must also be declared as dataclasses for their fields to be
    recognized by parse_cmd_line_args().
    """
    @staticmethod
    def _meta(help: str = '', required: bool = False):
        """Helper method to define metadata for dataclass fields."""
        if help != '':
            help = help + ' '
        return {
            'help': help + '(default: %(default)s)',
            'required': required,
        }

    loglevel: str = field(default='INFO',
                          metadata=_meta(help='Logging level for stdout logs (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)'))
    output_dir: Path = field(default=get_default_working_dir(),
                             metadata=_meta(help='Directory to save the data, checkpoints, trained model, etc. '
                                                 'A timestamped subdirectory will be created under this for each run.'))
    randseed: int = field(default=9_192_631_770,  # Frequency of ground state hyperfine transition of cesium-133 in Hz.
                          metadata=_meta(help='Random seed for reproducibility.'))
    epochs: int = field(default=10, metadata=_meta(help='Number of epochs to train the model.'))
    batch_size: int = field(default=8, metadata=_meta(help='Batch size for training.'))
    monitor_steps: int = field(default=100,
                               metadata=_meta(help='How frequently to log statistics to Tensorboard, in '
                                              'steps (minibatches). Set it to a negative number to disable logging.'))


def parse_cmd_line_args[T: BaseConfiguration](arg_template: T, description: Optional[str], argv: Optional[list[str]]) -> T:
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


def accuracy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simple accuracy for discrete predictions."""
    assert x.shape == y.shape
    return torch.sum(x == y, dim=0) / x.shape[0]


class SequenceCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """A version of CrossEntropy loss designed for sequential data.

    The default CrossEntropyLoss in torch is set up for multi-class, single-label classification. That
    is, it assumes that each instance in your batch is assigned a single category. That's a problem for
    sequence data, where a single "instance" is a sequence and each step in the sequence receives a
    category label. This class implements CrossEntropyLoss for data that is naturally sequence shaped.

    Specifically, the expected input to CrossEntropyLoss is of shape [batches, categories], but we have
    [batches, sequence_len, categories]. The usual answer is to reshape the data into
    [batches * sequence_len, categories] - i.e., to eliminate the sequence dimension and push it into
    batches. That's acceptable for a loss that averages over both batches and sequence (and doesn't, say,
    try to account for temporality within the sequence). This class is a simple wrapper that does that
    mapping transparently to the caller.
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute CrossEntropy loss for two sequence data tensors.

        Args:
            input (torch.Tensor): Model prediction. Shape: [batches, sequence_len, n_categories]
            target (torch.Tensor): Target value. Shape: [batches, n_categories]

        Returns:
            torch.Tensor: Cross Entropy loss between input and target.
        """
        n_categories = input.shape[-1]
        return super().forward(input.reshape(-1, n_categories), target.reshape(-1))


class BaseApp[T: BaseConfiguration](ABC):
    """Root of the App(lication) hierarchy.

    This abstract base class for the App hierarchy mostly sets up the
    baseline operating environment:
        - Parses command-line args.
        - Sets up logging and output.
        - Creates an output directory.
        - Saves a copy of the config to disk in the output dir.

    Type Parameters:
        T: Type of the command line arguments, which must be a subclass of BaseArguments.

    Arguments:
        arg_template (T): An instance of type T that will be filled with command-line data.
        description (str): A program description that will be printed with --help.
        argv (list[str]): The CLI arguments to the script. Available here as a test
            injection point.
    """
    def __init__(self, arg_template: T, description: str | None, argv: list[str] | None = None):
        self.config = parse_cmd_line_args(arg_template=arg_template, description=description, argv=argv)
        self.run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        self.work_dir = self.config.output_dir / self.run_timestamp
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logdir = self.work_dir / 'logs'
        self.logger = setup_logging(self.config.loglevel, self.logdir)
        config_dest = (self.work_dir / 'config.json')
        config_dest.write_text(json.dumps(asdict(self.config), indent=2, default=str))
        self.logger.debug('Saved config', file=str(config_dest))

    @abstractmethod
    def run(self):
        raise NotImplementedError('run() method needs to be defined by a concrete subclass')


class TrainableModelApp[T: BaseConfiguration, M: torch.nn.Module](BaseApp[T]):
    """Base class for applications that train models.

    This class provides a common interface for applications, including methods for running the application
    and setting up logging.

    Type Parameters:
        T: Type of the command line arguments, which must be a subclass of BaseArguments.
        M: Type of the model, which must be a subclass of torch.nn.Module.
    """
    def __init__(self, arg_template: T, description: str | None, argv: list[str] | None = None):
        super().__init__(arg_template, description, argv)
        self.model: Optional[M] = None
        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.debug('Assigned device', device=self.device)
        torch.manual_seed(self.config.randseed)
        self.logger.debug('Set random seed', randseed=self.config.randseed)
        self.tb_writer = SummaryWriter(log_dir=self.config.output_dir / 'tensorboard' / self.run_timestamp)

    def train_model(self,
                    data: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: torch.nn.Module) -> None:
        """Train the model.

        This is a generic training loop, appropriate for supervised model training.

        Requires: self.model has already been initialized.

        Arguments:
            data (DataLoader): DataLoader providing the training data. Assumes that each
                batch is a tuple (inputs, targets).
            optimizer (torch.optim.Optimizer): The optimizer to use for training. Must be
                initialized with the model's parameters and pre-configured with learning
                rate and other hyperparameters.
            loss_fn (torch.nn.Module): The loss function to use for training. Must be
                pre-configured (e.g., with reduction method).
            num_epochs (int): The number of epochs to train the model for.
        """
        assert self.model is not None, 'Model must be initialized before training.'
        self.logger.info('Starting model training', epochs=self.config.epochs)
        self.logger.debug('Optimizer', optimizer=optimizer)
        self.logger.debug('Loss function', loss_function=loss_fn)
        self.model.train()  # Set the model to training mode
        running_loss = 0.0
        running_steps = 0
        for epoch in tqdm.tqdm(range(self.config.epochs), desc='Epoch'):
            epoch_logger = self.logger.bind(epoch=epoch)
            for batch_id, batch_data in tqdm.tqdm(enumerate(data), desc='Batch', leave=False, total=len(data)):
                optimizer.zero_grad()  # Clear gradients
                predicted = self.model(*batch_data[:-1])
                for idx, d in enumerate(batch_data):
                    self.logger.debug('batch data', idx=idx, shape=d.shape)
                self.logger.debug('predicted', shape=predicted.shape)
                train_loss = loss_fn(predicted, batch_data[-1])
                train_loss.backward()
                optimizer.step()
                running_loss += train_loss.item()
                running_steps += 1
                if self.config.monitor_steps > 0 and batch_id % self.config.monitor_steps == 0:
                    global_step = epoch * len(data) + batch_id
                    epoch_logger.debug('Batch', batch=batch_id, global_step=global_step, loss=train_loss.item())
                    self.tb_writer.add_scalar('train loss',
                                              running_loss / running_steps,
                                              global_step=global_step)
                    for name, param in self.model.named_parameters():
                        self.tb_writer.add_histogram(name,
                                                     param,
                                                     global_step=global_step)
                    running_loss = 0.0
                    running_steps = 0
        self.logger.info('Model training completed')


def save_tensor(tensor: torch.Tensor, path: Path):
    """Save a tensor to both pt and txt files."""
    # TODO(heather): Add tests for this function.
    # TODO(heather): Move to safetenors representation.
    torch.save(tensor, path.with_suffix('.pt'))
    with open(path.with_suffix('.txt'), 'w') as f:
        f.write(pprint.pformat(tensor.tolist(), indent=2, width=80))
