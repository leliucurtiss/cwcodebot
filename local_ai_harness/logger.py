import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str | Path | None = None):
    """
    Configures the logging system.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    
    # Set a default logger for the application
    logger = logging.getLogger("local_ai_harness")
    return logger
