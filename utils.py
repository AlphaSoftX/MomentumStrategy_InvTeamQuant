import logging
from pathlib import Path

_LEVEL_RGB = {
    logging.DEBUG: (0, 255, 0),
    logging.INFO: (0, 0, 255),
    logging.WARNING: (255, 255, 0),
    logging.ERROR: (255, 100, 0),
    logging.CRITICAL: (255, 0, 0),
}

class LoggerFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(message)s")

    def format(self, record):
        r, g, b = _LEVEL_RGB.get(record.levelno, (255, 255, 255))
        return f"\033[38;2;{r};{g};{b}m[{record.name}]\033[0m {super().format(record)}"

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(LoggerFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger

def _safe_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "-").replace("^", "IDX_").replace(":", "_")
