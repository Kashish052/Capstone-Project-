"""Application logging with isolated, test-safe handlers."""
from pathlib import Path
import logging
import os

BASE_DIR = Path(__file__).resolve().parents[1]


def get_logger(name="aavail", log_dir=None):
    """Return a file-backed logger and avoid stale handlers across tests."""
    target_dir = Path(log_dir or os.getenv("LOG_DIR", BASE_DIR / "logs"))
    target_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    target_path = target_dir / "application.log"

    for handler in logger.handlers[:]:
        current = getattr(handler, "baseFilename", None)
        if current and Path(current).resolve() != target_path.resolve():
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

    if not any(
        getattr(handler, "baseFilename", None)
        and Path(handler.baseFilename).resolve() == target_path.resolve()
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(target_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)

    return logger
