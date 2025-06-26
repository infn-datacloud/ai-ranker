from unittest.mock import MagicMock, patch

import src.main as main


@patch("src.main.inference_run")
@patch("src.main.training_run")
@patch("src.main.create_logger")
@patch("src.parser.parser.parse_args")
def test_training_run_called(
    mock_parse_args, mock_create_logger, mock_training_run, mock_inference_run
):
    mock_parse_args.return_value = MagicMock(
        training=True, inference=False, loglevel="info"
    )
    logger = MagicMock()
    mock_create_logger.return_value = logger

    main.main()

    mock_training_run.assert_called_once()
    args_passed = mock_training_run.call_args[0]
    assert args_passed[0] is logger


@patch("src.main.inference_run")
@patch("src.main.training_run")
@patch("src.main.create_logger")
@patch("src.parser.parser.parse_args")
def test_inference_run_called(
    mock_parse_args, mock_create_logger, mock_training_run, mock_inference_run
):
    mock_parse_args.return_value = MagicMock(
        training=False, inference=True, loglevel="debug"
    )
    logger = MagicMock()
    mock_create_logger.return_value = logger

    main.main()

    mock_inference_run.assert_called_once()
    args_passed = mock_inference_run.call_args[0]
    assert args_passed[0] is logger


@patch("src.main.inference_run")
@patch("src.main.training_run")
@patch("src.main.create_logger")
@patch("src.parser.parser.parse_args")
def test_warning_logged_if_no_flags(
    mock_parse_args, mock_create_logger, mock_training_run, mock_inference_run
):
    mock_parse_args.return_value = MagicMock(
        training=False, inference=False, loglevel="warning"
    )
    logger = MagicMock()
    mock_create_logger.return_value = logger

    main.main()

    logger.warning.assert_called_once_with(
        "Neither --training or --inference arguments have been defined"
    )
