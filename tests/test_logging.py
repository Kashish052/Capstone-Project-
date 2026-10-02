import tempfile
import unittest
from pathlib import Path

from src.log import get_logger


class LoggingTests(unittest.TestCase):
    def test_logger_writes_to_isolated_directory(self):
        logger_name = f"test-{id(self)}"

        with tempfile.TemporaryDirectory() as tmp:
            logger = get_logger(logger_name, tmp)
            logger.info("unit-test-message")

            for handler in logger.handlers:
                handler.flush()

            log_file = Path(tmp) / "application.log"
            self.assertTrue(log_file.exists())
            self.assertIn(
                "unit-test-message",
                log_file.read_text(encoding="utf-8"),
            )

            # Windows keeps open file handles locked, so close them before
            # TemporaryDirectory removes the test directory.
            for handler in logger.handlers[:]:
                handler.flush()
                handler.close()
                logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
