import logging
import os
from datetime import datetime


def setup_logger(log_dir="logs", name="training_logger"):
    """
    Create and configure a logger.

    Logs are written to both file and console.
    """

    """
    Create log directory if it does not exist.
    """
    os.makedirs(log_dir, exist_ok=True)

    """
    Create logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    """
    Create unique log file name using current time.
    """
    currtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"training_{currtime}.log")

    """
    File handler for saving logs to file.
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    """
    Console handler for printing logs to terminal.
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    """
    Define log format.
    """
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    """
    Attach formatter to handlers.
    """
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    """
    Add handlers to logger.
    """
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
