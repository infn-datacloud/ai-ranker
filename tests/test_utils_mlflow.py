import base64
import pickle
from unittest.mock import MagicMock, patch

import mlflow
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
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


@patch(
    "mlflow.set_tracking_uri", side_effect=mlflow.exceptions.MlflowException("error")
)
@patch("src.settings.load_mlflow_settings")
def test_setup_mlflow_failure(mock_load_settings, mock_set_uri):
    logger = MagicMock()
    mock_settings = MagicMock()
    mock_settings.MLFLOW_HTTP_REQUEST_TIMEOUT = 10
    mock_settings.MLFLOW_HTTP_REQUEST_MAX_RETRIES = 3
    mock_settings.MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR = 1.0
    mock_settings.MLFLOW_HTTP_REQUEST_BACKOFF_JITTER = 0.1
    mock_settings.MLFLOW_TRACKING_URI = "http://mock_uri"
    mock_settings.MLFLOW_EXPERIMENT_NAME = "test_exp"
    mock_load_settings.return_value = mock_settings

    with pytest.raises(SystemExit):
        setup_mlflow(logger=logger)
    logger.error.assert_called_once()


@patch("mlflow.start_run", side_effect=mlflow.exceptions.MlflowException("fail"))
def test_log_on_mlflow_failure(mock_start_run):
    logger = MagicMock()
    model_params = {"param": 1}
    model_name = "TestModel"
    model = MagicMock()
    metrics = {"accuracy": 0.95}
    metadata = MagicMock()
    metadata.model_dump.return_value = {"author": "me"}
    df = pd.DataFrame({"feature": [1], "importance": [0.9]})
    scaler_file = "scaler.pkl"
    scaler_bytes = b"dummy"

    with pytest.raises(SystemExit):
        log_on_mlflow(
            model_params=model_params,
            model_name=model_name,
            model=model,
            metrics=metrics,
            metadata=metadata,
            feature_importance_df=df,
            scaling_enable=True,
            scaler_file=scaler_file,
            scaler_bytes=scaler_bytes,
            logger=logger,
        )
    logger.error.assert_called_once()


@patch("src.utils.mlflow.mlflow")
def test_log_on_mlflow_scaling_disabled(mock_mlflow, mock_logger, mock_metadata):
    model = LogisticRegression()
    model_params = {"C": 1.0}
    metrics = {"accuracy": 0.9}
    feature_importance_df = pd.DataFrame({"feature": ["a"], "importance": [0.5]})

    log_on_mlflow(
        model_params=model_params,
        model_name="TestModel",
        model=model,
        metrics=metrics,
        metadata=mock_metadata,
        feature_importance_df=feature_importance_df,
        scaling_enable=False,
        scaler_file=None,
        scaler_bytes=None,
        logger=mock_logger,
    )

    assert not any(
        call.args[0] == "scaler.pkl" for call in mock_mlflow.log_artifact.call_args_list
    )


@patch("src.utils.mlflow.mlflow.artifacts.load_dict")
def test_get_scaler_missing_key(mock_load_dict):
    mock_load_dict.return_value = {}

    with pytest.raises(KeyError, match="scaler.*not found"):
        get_scaler(model_uri="some/uri", scaler_file="scaler.pkl")


@patch(
    "src.utils.mlflow.mlflow.artifacts.load_dict", side_effect=Exception("load failed")
)
def test_get_scaler_load_dict_exception(mock_load_dict):
    with pytest.raises(Exception, match="load failed"):
        get_scaler(model_uri="some/uri", scaler_file="scaler.pkl")


@patch("src.utils.mlflow.mlflow.artifacts.load_dict")
def test_get_scaler_invalid_pickle(mock_load_dict):
    invalid_data = base64.b64encode(b"not a pickled object").decode("utf-8")
    mock_load_dict.return_value = {"scaler": invalid_data}

    with pytest.raises(Exception):
        get_scaler(model_uri="some/uri", scaler_file="scaler.pkl")


def test_get_model_uri_raises_value_error():
    client = MagicMock()
    client.search_model_versions.return_value = []

    with pytest.raises(ValueError, match=r"Model .* not found"):
        get_model_uri(
            model_name="modello_inesistente", model_version="9999", client=client
        )


def test_get_model_uri_raises_value_error_version_not_found():
    client = MagicMock()

    client.search_model_versions.return_value = [
        MagicMock(version=1),
        MagicMock(version=2),
    ]

    with pytest.raises(ValueError, match=r"Version .* not found"):
        get_model_uri(model_name="modello_esistente", model_version=9999, client=client)


def test_get_model_raises_value_error_on_load_failure():
    fake_uri = "fake_uri"

    mock_model = MagicMock()
    mock_model.metadata.flavors.keys.return_value = {"sklearn"}
    mock_model.loader_module = "mlflow.sklearn"

    with (
        patch("src.utils.mlflow.load_model", return_value=mock_model),
        patch("mlflow.sklearn.load_model") as mock_sklearn_load,
    ):
        mock_sklearn_load.side_effect = Exception("Errore simulato")

        with pytest.raises(
            ValueError, match=f"Model not found at given uri '{fake_uri}'"
        ):
            get_model(model_uri=fake_uri)


def test_get_model_raises_value_error_for_wrong_loader_module():
    fake_uri = "fake_uri"
    mock_model = MagicMock()
    mock_model.metadata.flavors.keys.return_value = {"sklearn"}
    mock_model.loader_module = "mlflow.pytorch"

    with patch("src.utils.mlflow.load_model", return_value=mock_model):
        with pytest.raises(
            ValueError, match="Model .* not in the mlflow.sklearn library"
        ):
            get_model(model_uri=fake_uri)


@patch("src.settings.MLFlowSettings", autospec=True)
@patch("mlflow.set_tracking_uri", side_effect=MlflowException("Test error"))
def test_setup_mlflow_exception_handling(mock_set_tracking_uri, mock_load_settings):
    logger = MagicMock()

    mock_settings = MagicMock()
    mock_settings.MLFLOW_HTTP_REQUEST_TIMEOUT = 10
    mock_settings.MLFLOW_HTTP_REQUEST_MAX_RETRIES = 3
    mock_settings.MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR = 0.5
    mock_settings.MLFLOW_HTTP_REQUEST_BACKOFF_JITTER = 0.1
    mock_settings.MLFLOW_TRACKING_URI = "http://localhost:5000"
    mock_settings.MLFLOW_EXPERIMENT_NAME = "test-exp"
    mock_load_settings.return_value = mock_settings

    with pytest.raises(SystemExit) as exc_info:
        setup_mlflow(logger=logger)

    assert exc_info.value.code == 1
    logger.error.assert_called_once()

    called_arg = logger.error.call_args[0][0]
    assert "Test error" in str(called_arg)
