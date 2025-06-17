import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.utils import (
    load_data_from_file,
    load_dataset_from_kafka_messages,
    load_local_dataset,
    write_data_to_file,
)

# === Costanti fittizie per i test ===
MSG_VERSION = "version"
MSG_VALID_KEYS = {"v1": {"feature1", "feature2"}}


# === Fixture comuni ===

@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def json_file(tmp_path):
    data = [{"feature1": 1, "feature2": 3}, {"feature1": 2, "feature2": 4}]
    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return str(file_path)


# === Test: load_local_dataset ===

def test_load_local_dataset_csv(csv_file, monkeypatch):
    monkeypatch.setattr("src.utils.MSG_VALID_KEYS", MSG_VALID_KEYS)
    logger = MagicMock()
    df = load_local_dataset(filename=csv_file, dataset_version="v1", logger=logger)
    assert not df.empty
    assert list(df.columns) == ["feature1", "feature2"]

def test_load_local_dataset_json(json_file, monkeypatch):
    monkeypatch.setattr("src.utils.MSG_VALID_KEYS", MSG_VALID_KEYS)
    logger = MagicMock()
    df = load_local_dataset(filename=json_file, dataset_version="v1", logger=logger)
    assert not df.empty
    assert list(df.columns) == ["feature1", "feature2"]

def test_load_local_dataset_invalid_extension(tmp_path):
    bad_file = tmp_path / "invalid.txt"
    bad_file.write_text("just text")
    logger = MagicMock()
    with pytest.raises(ValueError, match="Unsupported file extension: .txt"):
        load_local_dataset(filename=str(bad_file), dataset_version="v1", logger=logger)

def test_load_local_dataset_invalid_version(csv_file):
    logger = MagicMock()
    with pytest.raises(ValueError, match="Dataset version v999 not supported"):
        load_local_dataset(filename=csv_file, dataset_version="v999", logger=logger)


# === Test: load_dataset_from_kafka_messages ===

def test_load_dataset_from_kafka_messages(monkeypatch):
    monkeypatch.setattr("src.utils.MSG_VERSION", MSG_VERSION)
    monkeypatch.setattr("src.utils.MSG_VALID_KEYS", MSG_VALID_KEYS)

    consumer = [
        MagicMock(value={"version": "v1", "feature1": 1, "feature2": 2}),
        MagicMock(value={"version": "v1", "feature1": 3, "feature2": 4}),
    ]
    logger = MagicMock()
    df = load_dataset_from_kafka_messages(consumer=consumer, logger=logger)
    assert not df.empty
    assert list(df.columns) == ["feature1", "feature2"]

def test_load_dataset_from_kafka_invalid_keys(monkeypatch):
    monkeypatch.setattr("src.utils.MSG_VERSION", MSG_VERSION)
    monkeypatch.setattr("src.utils.MSG_VALID_KEYS", MSG_VALID_KEYS)

    consumer = [
        MagicMock(value={"version": "v1", "feature1": 1, "invalid_feature": 2}),
    ]
    logger = MagicMock()
    with pytest.raises(AssertionError, match="Found invalid keys"):
        load_dataset_from_kafka_messages(consumer=consumer, logger=logger)


# === Test: load_data_from_file ===

def test_load_data_from_file(tmp_path):
    file_path = tmp_path / "data.json"
    data = [{"key": 1}, {"key": 2}]
    file_path.write_text(json.dumps(data))

    logger = MagicMock()
    result = load_data_from_file(filename=str(file_path), logger=logger)
    assert result == data


# === Test: write_data_to_file ===

def test_write_data_to_file(tmp_path):
    file_path = tmp_path / "out.json"
    initial_data = [{"x": 1}]
    file_path.write_text(json.dumps(initial_data))

    new_entry = {"x": 2}
    logger = MagicMock()
    write_data_to_file(filename=str(file_path), data=new_entry, logger=logger)

    with open(file_path) as f:
        result = json.load(f)
    assert result == [{"x": 1}, {"x": 2}]
