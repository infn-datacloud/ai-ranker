import logging

import pytest

from src.parser import parser


def test_default_arguments():
    # Test that default values are set correctly
    args = parser.parse_args([])
    assert args.loglevel == "info"
    assert args.training is False
    assert args.inference is False

@pytest.mark.parametrize("loglevel", [lvl.lower() for lvl in logging._nameToLevel.keys()])
def test_valid_loglevels(loglevel):
    # Test that all valid log levels are accepted
    args = parser.parse_args(["--loglevel", loglevel])
    assert args.loglevel == loglevel

def test_training_flag():
    # Test that the training flag sets the correct attribute
    args = parser.parse_args(["--training"])
    assert args.training is True
    assert args.inference is False

def test_inference_flag():
    # Test that the inference flag sets the correct attribute
    args = parser.parse_args(["--inference"])
    assert args.inference is True
    assert args.training is False

def test_training_and_inference_flags():
    # Test that both flags can be set together
    args = parser.parse_args(["--training", "--inference"])
    assert args.training is True
    assert args.inference is True

def test_invalid_loglevel():
    # Test that an invalid log level causes a system exit
    with pytest.raises(SystemExit):
        parser.parse_args(["--loglevel", "invalidlevel"])

def test_short_flags():
    # Test short versions of the flags
    args = parser.parse_args(["-l", "warning", "-t", "-i"])
    assert args.loglevel == "warning"
    assert args.training is True
    assert args.inference is True
