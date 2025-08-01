"""Shared utility functions."""

import keyring  # Accessing API keys securely from Apple Keychain or Windows Credential Store.
from logging import basicConfig, getLogger, DEBUG, INFO

__all__ = ['setup_logging']

def setup_logging():
    """Set up logging configuration.

    Sets a local logger at DEBUG level and configures the logging format.
    """
    basicConfig(level=INFO,
                format='%(asctime)s|%(levelname)s|%(name)s> %(message)s',
                datefmt='%Y-%m-%dT%H:%M:%S%z')
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    return logger


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