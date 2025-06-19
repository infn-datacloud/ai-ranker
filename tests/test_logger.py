import logging
import sys
from unittest.mock import patch

from src.logger import StderrFilter, StdoutFilter, create_logger


def test_stdout_filter():
    f = StdoutFilter()
    record_warning = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    )
    record_error = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    )

    # WARNING level should pass the stdout filter
    assert f.filter(record_warning) is True
    # ERROR level should not pass the stdout filter
    assert f.filter(record_error) is False


def test_stderr_filter():
    f = StderrFilter()
    record_warning = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    )
    record_error = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    )

    # WARNING level should not pass the stderr filter
    assert f.filter(record_warning) is False
    # ERROR level should pass the stderr filter
    assert f.filter(record_error) is True


def test_create_logger_happy_path():
    logger = create_logger("test_logger", "INFO")
    # Check logger name
    assert logger.name == "test_logger"
    # Check logger level is INFO (numeric 20) or equivalent
    assert logger.level == logging.INFO or logger.level == 20
    # Logger should have two handlers
    assert len(logger.handlers) == 2

    stdout_handler = logger.handlers[0]
    stderr_handler = logger.handlers[1]

    # Check that stdout handler has StdoutFilter
    assert any(isinstance(f, StdoutFilter) for f in stdout_handler.filters)
    # Check that stderr handler has StderrFilter
    assert any(isinstance(f, StderrFilter) for f in stderr_handler.filters)

    # Check that both handlers have a formatter
    assert stdout_handler.formatter is not None
    assert stderr_handler.formatter is not None

    # Check that stdout handler writes to sys.stdout
    assert stdout_handler.stream == sys.stdout
    # The stderr handler uses default stream (usually sys.stderr)
    assert stderr_handler.stream is not None


def test_create_logger_invalid_level_logs_error(caplog):
    # Get the real logger first
    real_logger = logging.getLogger("test_logger")

    # Patch only the setLevel method of this logger to raise ValueError
    with patch.object(
        real_logger, "setLevel", side_effect=ValueError("Invalid level")
    ) as mock_set_level:
        with caplog.at_level(logging.ERROR):
            logger = create_logger("test_logger", "INVALID_LEVEL")

        # Check that error was logged on the real logger
        assert any("Invalid log level" in record.message for record in caplog.records)


def test_create_logger_with_level_none():
    logger = create_logger("test_logger")
    assert logger.name == "test_logger"
