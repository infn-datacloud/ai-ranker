import base64
import pickle
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from src.training.models import MetaData
from src.utils.mlflow import (
    get_model,
    get_model_uri,
    get_scaler,
    log_on_mlflow,
    setup_mlflow,
)


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_metadata():
    return MetaData(
        start_time="2024-01-01T00:00:00",
        end_time="2024-01-02T00:00:00",
        features=["a", "b", "c"],
        features_number=3,
        remove_outliers=False,
        scaling=True,
        scaler_file="scaler.pkl",
    )


@patch("src.utils.mlflow.mlflow")
@patch("src.utils.mlflow.load_mlflow_settings")
def test_setup_mlflow(mock_load_settings, mock_mlflow, mock_logger):
    mock_settings = MagicMock()
    mock_load_settings.return_value = mock_settings
    client = setup_mlflow(logger=mock_logger)
    assert mock_mlflow.set_tracking_uri.called
    assert mock_mlflow.set_experiment.called
    assert isinstance(client, MagicMock)


@patch("src.utils.mlflow.mlflow")
def test_log_on_mlflow(mock_mlflow, mock_logger, mock_metadata):
    model = LogisticRegression()
    model_params = {"C": 1.0}
    metrics = {"accuracy": 0.9}
    feature_importance_df = pd.DataFrame({"feature": ["a"], "importance": [0.5]})
    scaler_bytes = pickle.dumps(RobustScaler())
    scaler_file = "scaler.pkl"

    log_on_mlflow(
        model_params=model_params,
        model_name="TestModel",
        model=model,
        metrics=metrics,
        metadata=mock_metadata,
        feature_importance_df=feature_importance_df,
        scaling_enable=True,
        scaler_file=scaler_file,
        scaler_bytes=scaler_bytes,
        logger=mock_logger,
    )

    assert mock_mlflow.log_params.called
    assert mock_mlflow.log_metric.called
    assert mock_mlflow.sklearn.log_model.called
    assert mock_mlflow.set_tag.called


def test_get_model_uri_latest_version():
    client = MagicMock()

    mock_v1 = MagicMock()
    mock_v1.name = "MyModel"
    mock_v1.version = "1"

    mock_v2 = MagicMock()
    mock_v2.name = "MyModel"
    mock_v2.version = "2"

    client.search_model_versions.return_value = [mock_v1, mock_v2]

    uri = get_model_uri(client, model_name="MyModel", model_version="latest")
    assert uri == "models:/MyModel/2"


def test_get_model_uri_specific_version():
    client = MagicMock()

    # Definisci un mock con attributi reali
    v1 = MagicMock()
    v1.name = "MyModel"
    v1.version = "1"

    client.search_model_versions.return_value = [v1]

    uri = get_model_uri(client, model_name="MyModel", model_version=1)
    assert uri == "models:/MyModel/1"


def test_get_model_uri_not_found():
    client = MagicMock()
    client.search_model_versions.return_value = []
    with pytest.raises(ValueError, match="Model 'MyModel' not found"):
        get_model_uri(client, model_name="MyModel", model_version="latest")


@patch("src.utils.mlflow.load_model")
@patch("src.utils.mlflow.mlflow.sklearn.load_model")
def test_get_model(mock_sklearn_load_model, mock_load_model):
    model_uri = "models:/TestModel/1"
    mock_pyfunc_model = MagicMock()
    mock_pyfunc_model.loader_module = "mlflow.sklearn"
    mock_pyfunc_model.metadata.flavors.keys.return_value = ["sklearn"]
    mock_load_model.return_value = mock_pyfunc_model
    mock_sklearn_model = MagicMock()
    mock_sklearn_load_model.return_value = mock_sklearn_model

    pyfunc_model, sklearn_model = get_model(model_uri=model_uri)
    assert pyfunc_model == mock_pyfunc_model
    assert sklearn_model == mock_sklearn_model


@patch("src.utils.mlflow.mlflow.artifacts.load_dict")
def test_get_scaler(mock_load_dict):
    scaler = RobustScaler()
    scaler_bytes = pickle.dumps(scaler)
    encoded = base64.b64encode(scaler_bytes).decode("utf-8")
    mock_load_dict.return_value = {"scaler": encoded}

    result = get_scaler(model_uri="some/uri", scaler_file="scaler.pkl")
    assert isinstance(result, RobustScaler)
