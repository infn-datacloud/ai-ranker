import logging
import pytest
from src.parser import parser


def test_valid_loglevels_with_training():
    # Ensure all valid log levels are accepted when --training is specified
    for loglevel in logging._nameToLevel.keys():
        args = parser.parse_args(["--loglevel", loglevel.lower(), "--training"])
        assert args.loglevel == loglevel.lower()
        assert args.training is True
        assert args.inference is False


def test_training_flag_only():
    # Ensure --training alone is parsed correctly
    args = parser.parse_args(["--training"])
    assert args.training is True
    assert args.inference is False
    assert args.loglevel == "info"


def test_inference_flag_only():
    # Ensure --inference alone is parsed correctly
    args = parser.parse_args(["--inference"])
    assert args.inference is True
    assert args.training is False
    assert args.loglevel == "info"


def test_missing_required_flag():
    # No --training or --inference provided: should raise a SystemExit
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_both_training_and_inference():
    # Providing both --training and --inference should fail due to mutual exclusion
    with pytest.raises(SystemExit):
        parser.parse_args(["--training", "--inference"])


def test_invalid_loglevel():
    # Invalid log level should raise a SystemExit
    with pytest.raises(SystemExit):
        parser.parse_args(["--loglevel", "invalid", "--training"])


def test_short_flags_training():
    # Test short version of loglevel and training flags
    args = parser.parse_args(["-l", "warning", "-t"])
    assert args.loglevel == "warning"
    assert args.training is True
    assert args.inference is False


def test_short_flags_inference():
    # Test short version of loglevel and inference flags
    args = parser.parse_args(["-l", "error", "-i"])
    assert args.loglevel == "error"
    assert args.inference is True
    assert args.training is False
