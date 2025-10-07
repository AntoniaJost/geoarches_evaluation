import logging
import os

# emoji labels for message types to create better visability in logs
GREEN = "✅"
RED = "❌"
WARN = "⚠️"

def setup_logger(log_dir, level="DEBUG"):
    """
    Sets up a named logger 'cmor' that writes to logs/pipeline.log.
    If already set up, returns the existing logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "pipeline.log")

    # convert level string (e.g. "DEBUG") into numeric level so that level can be set within config file
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("cmor")
    logger.setLevel(numeric_level)

    # Only add handler if not already added
    if not logger.handlers:
        fh = logging.FileHandler(path)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

def success(logger, msg):
    logger.info(f"{GREEN} {msg}")

def error(logger, msg):
    logger.error(f"{RED} {msg}")

def warn(logger, msg):
    logger.warning(f"{WARN} {msg}")
