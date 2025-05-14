"""Custom logging module to format log messages with colors.

Adapted from the original code by Josh Cunningham (GitHub: @JoshCu)
https://github.com/CIROH-UA/NGIAB_data_preprocess

Adapted by Quinn Lee (GitHub @quinnylee)
"""

import logging

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages based on their level."""
    def format(self, record):
        message = super().format(record)
        if record.levelno == logging.DEBUG:
            return f"{Fore.BLUE}{message}{Style.RESET_ALL}"
        if record.levelno == logging.WARNING:
            return f"{Fore.YELLOW}{message}{Style.RESET_ALL}"
        if record.levelno == logging.CRITICAL or record.levelno == logging.ERROR:
            return f"{Fore.RED}{message}{Style.RESET_ALL}"
        if record.name == "root":  # Only color info messages from this script green
            return f"{Fore.GREEN}{message}{Style.RESET_ALL}"
        return message


def setup_logging(debug) -> None:
    """Set up logging configuration with green formatting."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    if debug:
        logging.basicConfig(level=logging.DEBUG, handlers=[handler])
    else:
        logging.basicConfig(level=logging.INFO, handlers=[handler])
