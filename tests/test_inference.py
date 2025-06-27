import base64
from logging import Logger
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from kafka.errors import NoBrokersAvailable
from sklearn.preprocessing import RobustScaler

import src.inference.main as inference
import src.utils.mlflow as mlflow_utils
from src.inference.main import (
    K_CLASS,
    K_REGR,
    K_RES_EXACT,
    MSG_TEMPLATE_NAME,
    MSG_VALID_KEYS,
    MSG_VERSION,
    InferenceSettings,
    connect_consumers_or_load_data,
    create_message,
    merge_and_sort_results,
    pre_process_message,
    run,
    send_message,
    sort_key,
    sort_key_with_exact_res,
)
from src.processing import (
    DF_AVG_FAIL_TIME,
    DF_AVG_SUCCESS_TIME,
    DF_FAIL_PERC,
    DF_MAX_DEP_TIME,
    DF_MIN_DEP_TIME,
    DF_PROVIDER,
    DF_TIMESTAMP,
    MSG_PROVIDER_NAME,
    MSG_REGION_NAME,
)
from src.utils import MSG_DEP_UUID, MSG_INSTANCE_REQ, MSG_INSTANCES_WITH_EXACT_FLAVORS


@pytest.fixture
def dummy_logger():
    # Logger mock that ignores all calls
    return MagicMock(spec=Logger)


@pytest.fixture
def dummy_mlflow_client():
    # Mock MLflow client
    return MagicMock()


@pytest.fixture
def dummy_model():
    # Mock sklearn model with required attributes and methods for prediction
    model = MagicMock()
    model.feature_names_in_ = ["feat1", "feat2"]
    model.predict_proba.return_value = np.array([[0.8, 0.2]])
    model.predict.return_value = np.array([5.0])
    return model


@pytest.fixture
def dummy_mlflow_model():
    # Mock MLflow model metadata for scaling and scaler_file
    mock_metadata = MagicMock()
    mock_metadata.metadata = {"scaling": True, "scaler_file": "scaler.pkl"}
    return MagicMock(metadata=mock_metadata)


@pytest.fixture
def dummy_scaler():
    # Mock scaler that simply returns the same array
    scaler = MagicMock(spec=RobustScaler)
    scaler.transform.side_effect = lambda x: x
    return scaler


def test_create_inference_input_without_scaler(dummy_model):
    # Test that DataFrame is created correctly without scaler applied
    data = {"feat1": 1.0, "feat2": 2.0, "extra": 5}
    df = inference.create_inference_input(data=data, model=dummy_model, scaler=None)
    assert list(df.columns) == ["feat1", "feat2"]
    assert df.iloc[0]["feat1"] == 1.0
    assert df.iloc[0]["feat2"] == 2.0


def test_create_inference_input_with_scaler(dummy_model, dummy_scaler):
    # Test that scaler.transform is applied when scaler is provided
    data = {"feat1": 3.0, "feat2": 4.0}
    df = inference.create_inference_input(
        data=data, model=dummy_model, scaler=dummy_scaler
    )
    dummy_scaler.transform.assert_called_once()
    assert list(df.columns) == ["feat1", "feat2"]


@patch("src.inference.main.get_model_uri")
@patch("src.inference.main.get_model")
@patch("src.inference.main.get_scaler")
def test_classification_predict_with_scaler(
    mock_get_scaler,
    mock_get_model,
    mock_get_uri,
    dummy_mlflow_client,
    dummy_logger,
    dummy_model,
    dummy_mlflow_model,
    dummy_scaler,
):
    # Test classification_predict loads model and scaler, predicts probabilities for providers
    mock_get_uri.return_value = "some_uri"
    mock_get_model.return_value = (dummy_mlflow_model, dummy_model)
    mock_get_scaler.return_value = dummy_scaler

    input_data = {
        "provider1": {"feat1": 1, "feat2": 2},
        "provider2": {"feat1": 3, "feat2": 4},
    }
    results = inference.classification_predict(
        input_data=input_data,
        model_name="model_name",
        model_version="1",
        mlflow_client=dummy_mlflow_client,
        logger=dummy_logger,
    )
    assert "provider1" in results
    assert "provider2" in results
    # The predicted success probability comes from the mocked model.predict_proba
    assert np.isclose(results["provider1"], 0.8)
    assert np.isclose(results["provider2"], 0.8)
    mock_get_uri.assert_called_once()
    mock_get_model.assert_called_once()
    mock_get_scaler.assert_called_once()


@patch("src.inference.main.get_model_uri")
@patch("src.inference.main.get_model")
def test_classification_predict_without_scaler(
    mock_get_model, mock_get_uri, dummy_mlflow_client, dummy_logger, dummy_model
):
    # Test classification_predict when scaler metadata flag is False (no scaler)
    dummy_mlflow_model = MagicMock()
    dummy_mlflow_model.metadata.metadata = {"scaling": False}
    mock_get_uri.return_value = "some_uri"
    mock_get_model.return_value = (dummy_mlflow_model, dummy_model)

    input_data = {
        "provider1": {"feat1": 1, "feat2": 2},
    }
    results = inference.classification_predict(
        input_data=input_data,
        model_name="model_name",
        model_version="1",
        mlflow_client=dummy_mlflow_client,
        logger=dummy_logger,
    )
    assert "provider1" in results
    assert np.isclose(results["provider1"], 0.8)
    mock_get_uri.assert_called_once()
    mock_get_model.assert_called_once()


@patch("src.inference.main.get_model_uri")
@patch("src.inference.main.get_model")
@patch("src.inference.main.get_scaler")
def test_regression_predict_with_scaler(
    mock_get_scaler,
    mock_get_model,
    mock_get_uri,
    dummy_mlflow_client,
    dummy_logger,
    dummy_model,
    dummy_mlflow_model,
    dummy_scaler,
):
    # Test regression_predict loads model and scaler, predicts times and calculates regression values clipped between 0 and 1
    mock_get_uri.return_value = "some_uri"
    mock_get_model.return_value = (dummy_mlflow_model, dummy_model)
    mock_get_scaler.return_value = dummy_scaler

    input_data = {
        "provider1": {
            "feat1": 1,
            "feat2": 2,
            inference.DF_MIN_DEP_TIME: 1.0,
            inference.DF_MAX_DEP_TIME: 10.0,
        },
        "provider2": {
            "feat1": 3,
            "feat2": 4,
            inference.DF_MIN_DEP_TIME: 5.0,
            inference.DF_MAX_DEP_TIME: 0.0,  # special case max=0
        },
    }
    results = inference.regression_predict(
        input_data=input_data,
        model_name="model_name",
        model_version="1",
        mlflow_client=dummy_mlflow_client,
        logger=dummy_logger,
    )
    assert "provider1" in results
    assert 0 <= results["provider1"] <= 1
    assert results["provider2"] == 1  # because max_regression_time == 0


@patch("src.inference.main.get_model_uri")
@patch("src.inference.main.get_model")
def test_regression_predict_without_scaler(
    mock_get_model, mock_get_uri, dummy_mlflow_client, dummy_logger, dummy_model
):
    # Test regression_predict when no scaler is loaded (scaling=False)
    dummy_mlflow_model = MagicMock()
    dummy_mlflow_model.metadata.metadata = {"scaling": False}
    mock_get_uri.return_value = "some_uri"
    mock_get_model.return_value = (dummy_mlflow_model, dummy_model)

    input_data = {
        "provider1": {
            "feat1": 1,
            "feat2": 2,
            inference.DF_MIN_DEP_TIME: 1.0,
            inference.DF_MAX_DEP_TIME: 10.0,
        }
    }
    results = inference.regression_predict(
        input_data=input_data,
        model_name="model_name",
        model_version="1",
        mlflow_client=dummy_mlflow_client,
        logger=dummy_logger,
    )
    assert "provider1" in results
    assert 0 <= results["provider1"] <= 1


def test_predict_success(monkeypatch, dummy_mlflow_client, dummy_logger):
    # Test predict returns classification and regression results without exception
    def dummy_classification_predict(**kwargs):
        return {"p1": 0.7, "p2": 0.6}

    def dummy_regression_predict(**kwargs):
        return {"p1": 0.4, "p2": 0.5}

    monkeypatch.setattr(
        inference, "classification_predict", dummy_classification_predict
    )
    monkeypatch.setattr(inference, "regression_predict", dummy_regression_predict)

    settings = InferenceSettings()
    settings.CLASSIFICATION_MODEL_NAME = "cmodel"
    settings.CLASSIFICATION_MODEL_VERSION = "1"
    settings.REGRESSION_MODEL_NAME = "rmodel"
    settings.REGRESSION_MODEL_VERSION = "1"

    input_inference = {"p1": {}, "p2": {}}
    classification, regression = inference.predict(
        input_inference=input_inference,
        mlflow_client=dummy_mlflow_client,
        settings=settings,
        logger=dummy_logger,
    )
    assert classification.keys() == regression.keys() == input_inference.keys()


def test_predict_handles_value_error(monkeypatch, dummy_mlflow_client, dummy_logger):
    # Test predict handles ValueError exceptions in classification and regression and returns NO_PREDICTED_VALUE
    def raise_value_error(*args, **kwargs):
        raise ValueError("Test error")

    monkeypatch.setattr(inference, "classification_predict", raise_value_error)
    monkeypatch.setattr(inference, "regression_predict", raise_value_error)

    settings = InferenceSettings()
    settings.CLASSIFICATION_MODEL_NAME = "cmodel"
    settings.CLASSIFICATION_MODEL_VERSION = "1"
    settings.REGRESSION_MODEL_NAME = "rmodel"
    settings.REGRESSION_MODEL_VERSION = "1"

    input_inference = {"p1": {}, "p2": {}}
    classification, regression = inference.predict(
        input_inference=input_inference,
        mlflow_client=dummy_mlflow_client,
        settings=settings,
        logger=dummy_logger,
    )
    for val in classification.values():
        assert val == inference.NO_PREDICTED_VALUE
    for val in regression.values():
        assert val == inference.NO_PREDICTED_VALUE


def test_get_model_uri_returns_uri():
    dummy_mlflow_client = MagicMock()

    mock_version = MagicMock()
    mock_version.name = "test_model"
    mock_version.version = "1"

    dummy_mlflow_client.search_model_versions.return_value = [mock_version]

    uri = mlflow_utils.get_model_uri(
        client=dummy_mlflow_client, model_name="test_model", model_version="1"
    )

    assert uri == "models:/test_model/1"


@patch("src.utils.mlflow.load_model")
@patch("src.utils.mlflow.mlflow.sklearn.load_model")
def test_get_model_loads_model_and_metadata(mock_sklearn_load_model, mock_load_model):
    # Setup mock per il modello MLflow e sklearn
    mock_pyfunc_model = MagicMock()
    # mock metadata.flavors.keys() to include "sklearn"
    mock_pyfunc_model.metadata.flavors.keys.return_value = ["sklearn"]
    mock_pyfunc_model.loader_module = "mlflow.sklearn"

    mock_load_model.return_value = mock_pyfunc_model
    mock_sklearn_load_model.return_value = "sklearn_model_instance"

    pyfunc_model, sklearn_model = mlflow_utils.get_model(model_uri="dummy_uri")

    # Assert
    assert pyfunc_model == mock_pyfunc_model
    assert sklearn_model == "sklearn_model_instance"
    mock_load_model.assert_called_once_with("dummy_uri")
    mock_sklearn_load_model.assert_called_once_with("dummy_uri")


@patch("src.utils.mlflow.load_model")
def test_get_model_raises_value_error_on_wrong_loader(mock_load_model):
    # Case when loader_module is not sklearn
    mock_model = MagicMock()
    mock_model.metadata.flavors.keys.return_value = ["other_flavor"]
    mock_model.loader_module = "mlflow.pytorch"
    mock_load_model.return_value = mock_model

    with pytest.raises(ValueError, match="not in the mlflow.sklearn library"):
        mlflow_utils.get_model(model_uri="dummy_uri")


@patch("src.utils.mlflow.load_model")
@patch("src.utils.mlflow.mlflow.sklearn.load_model")
def test_get_model_raises_value_error_on_sklearn_load_failure(
    mock_sklearn_load_model, mock_load_model
):
    mock_model = MagicMock()
    mock_model.metadata.flavors.keys.return_value = ["sklearn"]
    mock_model.loader_module = "mlflow.sklearn"
    mock_load_model.return_value = mock_model

    mock_sklearn_load_model.side_effect = Exception("load failure")

    with pytest.raises(ValueError, match="Model not found at given uri"):
        mlflow_utils.get_model(model_uri="dummy_uri")


@patch("src.utils.mlflow.mlflow.artifacts.load_dict")
@patch("pickle.loads")
def test_get_scaler_loads_scaler_correctly(mock_pickle_loads, mock_load_dict):
    scaler_bytes = b"dummy scaler bytes"
    encoded_scaler = base64.b64encode(scaler_bytes).decode()

    mock_load_dict.return_value = {"scaler": encoded_scaler}
    mock_scaler = RobustScaler()
    mock_pickle_loads.return_value = mock_scaler

    scaler = mlflow_utils.get_scaler(model_uri="model_uri", scaler_file="scaler.pkl")

    assert scaler == mock_scaler
    mock_load_dict.assert_called_once_with("model_uri/scaler.pkl")
    mock_pickle_loads.assert_called_once_with(scaler_bytes)


@patch("src.utils.mlflow.mlflow.artifacts.load_dict")
def test_get_scaler_raises_key_error_if_no_scaler_key(mock_load_dict):
    mock_load_dict.return_value = {"not_scaler": "data"}

    with pytest.raises(KeyError, match="'scaler' key not found"):
        mlflow_utils.get_scaler(model_uri="model_uri", scaler_file="scaler.pkl")


def test_create_inference_input_with_missing_features(dummy_model):
    """
    Test that create_inference_input handles input data missing model feature keys gracefully.
    It should only include features that model expects, so here input has extra and missing keys.
    """
    data = {"feat1": 10, "unwanted_feat": 99}  # missing "feat2"
    # The DataFrame should have columns only feat1, feat2 (feat2 missing in data will be NaN)
    df = inference.create_inference_input(data=data, model=dummy_model, scaler=None)
    # The resulting DataFrame must have exactly the model's features columns
    assert list(df.columns) == dummy_model.feature_names_in_
    # feat1 should have the value provided
    assert df.iloc[0]["feat1"] == 10
    # feat2 is missing in data, so should be NaN
    assert np.isnan(df.iloc[0]["feat2"])


def test_create_inference_input_with_empty_data_and_scaler(dummy_model, dummy_scaler):
    """
    Test create_inference_input when empty data is provided.
    The scaler should still be called but on empty DataFrame.
    """
    data = {}
    df = inference.create_inference_input(
        data=data, model=dummy_model, scaler=dummy_scaler
    )
    # The resulting df should have correct columns but zero rows (empty)
    assert list(df.columns) == dummy_model.feature_names_in_
    assert (
        df.shape[0] == 1
    )  # Usually create_inference_input creates a 1-row df even if data empty
    dummy_scaler.transform.assert_called_once()


@patch("src.inference.main.get_model_uri")
@patch("src.inference.main.get_model")
@patch("src.inference.main.get_scaler")
def test_classification_predict_raises_value_error_on_get_model(
    mock_get_scaler, mock_get_model, mock_get_uri, dummy_mlflow_client, dummy_logger
):
    """
    Test classification_predict when get_model raises ValueError.
    It should propagate the exception.
    """
    mock_get_uri.return_value = "uri"
    mock_get_model.side_effect = ValueError("model error")
    mock_get_scaler.return_value = None

    with pytest.raises(ValueError, match="model error"):
        inference.classification_predict(
            input_data={"p": {"feat1": 1, "feat2": 2}},
            model_name="name",
            model_version="1",
            mlflow_client=dummy_mlflow_client,
            logger=dummy_logger,
        )


@patch("src.inference.main.get_model_uri")
@patch("src.inference.main.get_model")
@patch("src.inference.main.get_scaler")
def test_regression_predict_raises_value_error_on_get_model(
    mock_get_scaler, mock_get_model, mock_get_uri, dummy_mlflow_client, dummy_logger
):
    """
    Test regression_predict when get_model raises ValueError.
    It should propagate the exception.
    """
    mock_get_uri.return_value = "uri"
    mock_get_model.side_effect = ValueError("model error")
    mock_get_scaler.return_value = None

    with pytest.raises(ValueError, match="model error"):
        inference.regression_predict(
            input_data={
                "p": {
                    "feat1": 1,
                    "feat2": 2,
                    inference.DF_MIN_DEP_TIME: 1.0,
                    inference.DF_MAX_DEP_TIME: 2.0,
                }
            },
            model_name="name",
            model_version="1",
            mlflow_client=dummy_mlflow_client,
            logger=dummy_logger,
        )


def test_predict_returns_empty_dicts_on_empty_input(
    monkeypatch, dummy_mlflow_client, dummy_logger
):
    """
    Test that predict returns empty dicts when input_inference is empty.
    """
    monkeypatch.setattr(inference, "classification_predict", lambda **kwargs: {})
    monkeypatch.setattr(inference, "regression_predict", lambda **kwargs: {})

    settings = inference.InferenceSettings()
    classification, regression = inference.predict(
        input_inference={},
        mlflow_client=dummy_mlflow_client,
        settings=settings,
        logger=dummy_logger,
    )
    assert classification == {}
    assert regression == {}


@patch("src.utils.mlflow.mlflow.artifacts.load_dict")
@patch("pickle.loads")
def test_get_scaler_raises_key_error_and_not_found(mock_pickle_loads, mock_load_dict):
    """
    Test get_scaler raises KeyError when 'scaler' key is missing in loaded dict.
    Also test the case where load_dict returns empty dict.
    """
    mock_load_dict.return_value = {}
    with pytest.raises(KeyError):
        mlflow_utils.get_scaler(model_uri="uri", scaler_file="scaler.pkl")


@patch("src.utils.mlflow.mlflow.artifacts.load_dict")
@patch("pickle.loads")
def test_get_scaler_raises_exception_on_pickle_load(mock_pickle_loads, mock_load_dict):
    """
    Test get_scaler raises if pickle.loads throws an Exception.
    """
    scaler_bytes = b"bad bytes"
    encoded_scaler = base64.b64encode(scaler_bytes).decode()
    mock_load_dict.return_value = {"scaler": encoded_scaler}
    mock_pickle_loads.side_effect = Exception("pickle error")

    with pytest.raises(Exception, match="pickle error"):
        mlflow_utils.get_scaler(model_uri="uri", scaler_file="scaler.pkl")


def test_sort_key_with_exact_res_prioritizes_resource_exactness():
    # Create two items: one with higher resource exactness but lower class+regr sum,
    # and another with lower resource exactness but higher sum
    item1 = (
        "provA",
        {K_RES_EXACT: 0.9, K_CLASS: 0.2, K_REGR: 0.1},
    )  # res_exact=0.9, sum=0.3
    item2 = (
        "provB",
        {K_RES_EXACT: 0.5, K_CLASS: 0.6, K_REGR: 0.4},
    )  # res_exact=0.5, sum=1.0

    # sort_key_with_exact_res returns a tuple: (res_exact, sum)
    key1 = sort_key_with_exact_res(item1)
    key2 = sort_key_with_exact_res(item2)

    # We expect key1 > key2 because res_exact is prioritized
    assert key1 > key2
    assert key1[0] == pytest.approx(0.9)
    assert key1[1] == pytest.approx(0.3)
    assert key2[0] == pytest.approx(0.5)
    assert key2[1] == pytest.approx(1.0)


def test_sort_key_sorts_by_combined_score():
    # Create two items with same res_exact but different sums
    item_low = ("provX", {K_RES_EXACT: 0.7, K_CLASS: 0.1, K_REGR: 0.2})  # sum=0.3
    item_high = ("provY", {K_RES_EXACT: 0.7, K_CLASS: 0.4, K_REGR: 0.6})  # sum=1.0

    # sort_key returns only the sum
    key_low = sort_key(item_low)
    key_high = sort_key(item_high)

    assert key_high > key_low
    assert key_low == pytest.approx(0.3)
    assert key_high == pytest.approx(1.0)


def test_sort_functions_edge_cases_zero_values():
    # Edge case: both values zero
    item_zero = ("provZ", {K_RES_EXACT: 0.0, K_CLASS: 0.0, K_REGR: 0.0})
    assert sort_key(item_zero) == 0.0
    assert sort_key_with_exact_res(item_zero) == (0.0, 0.0)


def test_sort_functions_negative_values():
    # Even though probabilities aren't negative in real world,
    # ensure functions handle negative values gracefully
    item_neg = ("provNeg", {K_RES_EXACT: -0.2, K_CLASS: -0.1, K_REGR: -0.3})
    assert sort_key(item_neg) == pytest.approx(-0.4)
    assert sort_key_with_exact_res(item_neg) == (-0.2, pytest.approx(-0.4))


@patch("src.inference.main.calculate_derived_properties")
def test_pre_process_message_with_no_historical_data(mock_calc):
    # Fake logger
    logger = MagicMock()

    # Simulate message version and valid keys
    version = "v1"
    MSG_VALID_KEYS[version] = {
        "provider",
        "region",
        MSG_INSTANCE_REQ,
        MSG_INSTANCES_WITH_EXACT_FLAVORS,
    }

    message = {
        MSG_VERSION: version,
        MSG_TEMPLATE_NAME: "templateA",
        "providers": [
            {
                "provider": "provA",
                "region": "eu",
                MSG_INSTANCE_REQ: 5,
                MSG_INSTANCES_WITH_EXACT_FLAVORS: 3,
            }
        ],
    }

    df = pd.DataFrame()  # empty historical data

    enriched_df = pd.DataFrame(
        [
            {
                DF_PROVIDER: "provA-eu",
                MSG_TEMPLATE_NAME: "templateA",
                MSG_INSTANCE_REQ: 5,
                MSG_INSTANCES_WITH_EXACT_FLAVORS: 3,
                MSG_PROVIDER_NAME: "provA",
                MSG_REGION_NAME: "eu",
            }
        ]
    )
    mock_calc.return_value = enriched_df

    result = pre_process_message(
        message=message.copy(), df=df, complex_templates=["templateA"], logger=logger
    )

    assert "provA-eu" in result
    entry = result["provA-eu"]
    assert entry[DF_AVG_SUCCESS_TIME] == 0.0
    assert entry[DF_FAIL_PERC] == 0.0
    assert entry[K_RES_EXACT] == 3 / 5

    logger.info.assert_called_once()
    logger.debug.assert_called_once()


def test_pre_process_message_invalid_version():
    logger = MagicMock()
    message = {
        MSG_VERSION: "unsupported_version",
        MSG_TEMPLATE_NAME: "templateA",
        "providers": [],
    }
    with pytest.raises(
        ValueError, match="Message version unsupported_version not supported"
    ):
        pre_process_message(
            message=message, df=pd.DataFrame(), complex_templates=[], logger=logger
        )


def test_pre_process_message_invalid_keys():
    version = "v1"
    MSG_VALID_KEYS[version] = {"provider", "region"}
    logger = MagicMock()
    message = {
        MSG_VERSION: version,
        MSG_TEMPLATE_NAME: "templateA",
        "providers": [{"provider": "provA", "region": "eu", "invalid_key": "bad"}],
    }
    with pytest.raises(AssertionError, match="Found invalid keys:"):
        pre_process_message(
            message=message, df=pd.DataFrame(), complex_templates=[], logger=logger
        )


def test_merge_no_filter_no_precedence():
    input_inf = {
        "provA": {K_RES_EXACT: 2},
        "provB": {K_RES_EXACT: 1},
    }
    class_resp = {"provA": 0.8, "provB": 0.6}
    regr_resp = {"provA": 0.2, "provB": 0.4}
    settings = InferenceSettings(
        CLASSIFICATION_WEIGHT=0.5,
        FILTER=False,
        THRESHOLD=0.0,
        EXACT_RESOURCES_PRECEDENCE=False,
    )
    logger = MagicMock()

    result = merge_and_sort_results(
        input_inference=input_inf,
        classification_response=class_resp,
        regression_response=regr_resp,
        settings=settings,
        logger=logger,
    )

    assert "provA" in result and "provB" in result
    valA = result["provA"]
    valB = result["provB"]
    assert valA[K_CLASS] == 0.8 * 0.5
    assert valA[K_REGR] == 0.2 * 0.5
    assert valA[K_RES_EXACT] == 2
    assert set(result.keys()) == {"provA", "provB"}


def test_merge_with_filter():
    input_inf = {
        "provA": {K_RES_EXACT: 2},
        "provB": {K_RES_EXACT: 1},
    }
    class_resp = {"provA": 0.9, "provB": 0.3}
    regr_resp = {"provA": 0.2, "provB": 0.1}
    settings = InferenceSettings(
        CLASSIFICATION_WEIGHT=0.6,
        FILTER=True,
        THRESHOLD=0.5,
        EXACT_RESOURCES_PRECEDENCE=False,
    )
    logger = MagicMock()

    result = merge_and_sort_results(
        input_inference=input_inf,
        classification_response=class_resp,
        regression_response=regr_resp,
        settings=settings,
        logger=logger,
    )

    assert "provA" in result
    assert "provB" not in result


def test_merge_with_exact_resources_precedence():
    input_inf = {
        "provA": {K_RES_EXACT: 0},
        "provB": {K_RES_EXACT: 3},
    }
    class_resp = {"provA": 0.5, "provB": 0.5}
    regr_resp = {"provA": 0.5, "provB": 0.5}
    settings = InferenceSettings(
        CLASSIFICATION_WEIGHT=0.5,
        FILTER=False,
        THRESHOLD=0.0,
        EXACT_RESOURCES_PRECEDENCE=True,
    )
    logger = MagicMock()

    result = merge_and_sort_results(
        input_inference=input_inf,
        classification_response=class_resp,
        regression_response=regr_resp,
        settings=settings,
        logger=logger,
    )

    keys = list(result.keys())
    assert keys[0] == "provB"
    assert keys[1] == "provA"


def test_merge_empty_inputs():
    settings = InferenceSettings()
    logger = MagicMock()
    result = merge_and_sort_results(
        input_inference={},
        classification_response={},
        regression_response={},
        settings=settings,
        logger=logger,
    )
    assert result == {}


def test_logger_called():
    input_inf = {
        "provA": {K_RES_EXACT: 1},
    }
    class_resp = {"provA": 1.0}
    regr_resp = {"provA": 0.0}
    settings = InferenceSettings()
    logger = MagicMock()

    merge_and_sort_results(
        input_inference=input_inf,
        classification_response=class_resp,
        regression_response=regr_resp,
        settings=settings,
        logger=logger,
    )
    assert logger.debug.called


def dummy_calculate_derived_properties(df, complex_templates):
    df[DF_PROVIDER] = df[MSG_PROVIDER_NAME] + "-" + df[MSG_REGION_NAME]
    return df


@patch(
    "src.inference.main.MSG_VALID_KEYS",
    new={
        "v1": {
            MSG_PROVIDER_NAME,
            MSG_REGION_NAME,
            MSG_INSTANCE_REQ,
            MSG_INSTANCES_WITH_EXACT_FLAVORS,
        }
    },
)
@patch(
    "src.inference.main.calculate_derived_properties",
    side_effect=dummy_calculate_derived_properties,
)
def test_pre_process_message_covers_all_rows(mock_calc):
    logger = MagicMock()
    message = {
        MSG_VERSION: "v1",
        MSG_TEMPLATE_NAME: "templateA",
        "providers": [
            {
                MSG_PROVIDER_NAME: "provA",
                MSG_REGION_NAME: "eu",
                MSG_INSTANCE_REQ: 5,
                MSG_INSTANCES_WITH_EXACT_FLAVORS: 3,
            }
        ],
    }

    df = pd.DataFrame(
        [
            {
                MSG_TEMPLATE_NAME: "templateA",
                DF_PROVIDER: "provA-eu",
                DF_TIMESTAMP: 1000,
                DF_AVG_SUCCESS_TIME: 10.0,
                DF_AVG_FAIL_TIME: 1.5,
                DF_FAIL_PERC: 0.3,
                DF_MIN_DEP_TIME: 5.0,
                DF_MAX_DEP_TIME: 20.0,
            },
            {
                MSG_TEMPLATE_NAME: "templateA",
                DF_PROVIDER: "provA-eu",
                DF_TIMESTAMP: 1100,
                DF_AVG_SUCCESS_TIME: 12.0,
                DF_AVG_FAIL_TIME: 1.0,
                DF_FAIL_PERC: 0.1,
                DF_MIN_DEP_TIME: 4.0,
                DF_MAX_DEP_TIME: 22.0,
            },
        ]
    )

    result = pre_process_message(
        message=message.copy(),
        df=df,
        complex_templates=["templateA"],
        logger=logger,
    )

    assert "provA-eu" in result

    assert result["provA-eu"][DF_AVG_SUCCESS_TIME] == 12.0
    assert result["provA-eu"][DF_AVG_FAIL_TIME] == 1.0
    assert result["provA-eu"][DF_FAIL_PERC] == 0.1
    assert result["provA-eu"][DF_MIN_DEP_TIME] == 4.0
    assert result["provA-eu"][DF_MAX_DEP_TIME] == 22.0

    # (3 / 5 = 0.6)
    assert result["provA-eu"][K_RES_EXACT] == 0.6

    logger.info.assert_not_called()
    assert logger.debug.call_count > 0


def test_create_message_success():
    logger = MagicMock()
    sorted_results = {
        "provA-eu": {"score": 0.9},
    }
    input_data = [
        {MSG_PROVIDER_NAME: "provA", MSG_REGION_NAME: "eu", "extra": 123},
    ]
    deployment_uuid = "uuid-1234"

    result = create_message(
        sorted_results=sorted_results,
        input_data=input_data,
        deployment_uuid=deployment_uuid,
        logger=logger,
    )
    assert result[MSG_DEP_UUID] == deployment_uuid
    assert len(result["ranked_providers"]) == 1
    assert result["ranked_providers"][0]["score"] == 0.9
    logger.debug.assert_called_once()


def test_create_message_no_match_raises():
    logger = MagicMock()
    sorted_results = {
        "provA-eu": {"score": 0.9},
        "provB-us": {"score": 0.7},
    }
    input_data = [
        {MSG_PROVIDER_NAME: "provA", MSG_REGION_NAME: "eu"},
    ]
    deployment_uuid = "uuid-1234"

    with pytest.raises(
        ValueError, match="No matching input_data entry for provider key 'provB-us'"
    ):
        create_message(
            sorted_results=sorted_results,
            input_data=input_data,
            deployment_uuid=deployment_uuid,
            logger=logger,
        )
    logger.error.assert_called_once_with(
        "No matching input_data entry for provider key 'provB-us'"
    )


@patch("src.inference.main.write_data_to_file")
def test_send_message_local_mode_no_file(mock_write_data):
    logger = MagicMock()
    settings = MagicMock(
        LOCAL_MODE=True,
        LOCAL_OUT_MESSAGES=None,
    )
    message = {"key": "value"}

    send_message(message, settings, logger)

    logger.error.assert_called_once_with(
        "LOCAL_OUT_MESSAGES environment variable has not been set."
    )
    mock_write_data.assert_not_called()


@patch("src.inference.main.write_data_to_file")
def test_send_message_local_mode_with_file(mock_write_data):
    logger = MagicMock()
    settings = MagicMock(
        LOCAL_MODE=True,
        LOCAL_OUT_MESSAGES="output.json",
    )
    message = {"key": "value"}

    send_message(message, settings, logger)

    mock_write_data.assert_called_once_with(
        filename="output.json", data=message, logger=logger
    )
    logger.error.assert_not_called()


@patch("src.inference.main.create_kafka_producer")
def test_send_message_kafka_mode(mock_create_producer):
    logger = MagicMock()
    settings = MagicMock(
        LOCAL_MODE=False,
        KAFKA_HOSTNAME="kafka:9092",
        KAFKA_RANKED_PROVIDERS_TOPIC="test-topic",
    )
    message = {"key": "value"}

    mock_producer = MagicMock()
    mock_create_producer.return_value = mock_producer

    send_message(message, settings, logger)

    mock_create_producer.assert_called_once_with(
        kafka_server_url="kafka:9092", logger=logger
    )
    mock_producer.send.assert_called_once_with("test-topic", message)
    mock_producer.close.assert_called_once()

    logger.info.assert_called_once_with(
        "Message sent to topic '%s' of kafka server'%s'",
        "test-topic",
        "kafka:9092",
    )


@patch("src.inference.main.load_data_from_file")
def test_local_mode_no_in_messages(mock_load_data):
    logger = MagicMock()
    settings = MagicMock(
        LOCAL_MODE=True,
        LOCAL_IN_MESSAGES=None,
    )
    with pytest.raises(SystemExit) as e:
        connect_consumers_or_load_data(settings, logger)
    logger.error.assert_called_once_with(
        "LOCAL_IN_MESSAGES environment variable has not been set."
    )
    assert e.value.code == 1
    mock_load_data.assert_not_called()


@patch("src.inference.main.load_data_from_file")
def test_local_mode_file_not_found(mock_load_data):
    logger = MagicMock()
    settings = MagicMock(
        LOCAL_MODE=True,
        LOCAL_IN_MESSAGES="infile.json",
        LOCAL_OUT_MESSAGES="outfile.json",
    )
    mock_load_data.side_effect = FileNotFoundError

    with pytest.raises(SystemExit) as e:
        connect_consumers_or_load_data(settings, logger)

    logger.error.assert_called_once_with("File '%s' not found", "infile.json")
    assert e.value.code == 1


@patch("src.inference.main.load_data_from_file")
def test_local_mode_success(mock_load_data):
    logger = MagicMock()
    settings = MagicMock(
        LOCAL_MODE=True,
        LOCAL_IN_MESSAGES="infile.json",
        LOCAL_OUT_MESSAGES="outfile.json",
    )
    mock_load_data.side_effect = [
        [{"input": 1}],
        [{"output": 2}],
    ]

    inputs, outputs = connect_consumers_or_load_data(settings, logger)
    assert inputs == [{"input": 1}]
    assert outputs == [{"output": 2}]
    assert logger.error.call_count == 0


@patch("src.inference.main.create_kafka_consumer")
def test_kafka_mode_success(mock_create_consumer):
    logger = MagicMock()
    settings = MagicMock(
        LOCAL_MODE=False,
        KAFKA_HOSTNAME="kafka:9092",
        KAFKA_INFERENCE_TOPIC="input-topic",
        KAFKA_INFERENCE_TOPIC_TIMEOUT=1000,
        KAFKA_RANKED_PROVIDERS_TOPIC="output-topic",
        KAFKA_RANKED_PROVIDERS_TOPIC_TIMEOUT=2000,
    )
    mock_input_consumer = MagicMock()
    mock_output_consumer = MagicMock()

    mock_create_consumer.side_effect = [mock_input_consumer, mock_output_consumer]

    input_consumer, output_consumer = connect_consumers_or_load_data(settings, logger)
    assert input_consumer == mock_input_consumer
    assert output_consumer == mock_output_consumer

    mock_create_consumer.assert_any_call(
        kafka_server_url="kafka:9092",
        topic="input-topic",
        consumer_timeout_ms=1000,
        logger=logger,
    )
    mock_create_consumer.assert_any_call(
        kafka_server_url="kafka:9092",
        topic="output-topic",
        consumer_timeout_ms=2000,
        logger=logger,
    )


@patch("src.inference.main.create_kafka_consumer")
def test_kafka_mode_no_brokers(mock_create_consumer):
    logger = MagicMock()
    settings = MagicMock(
        LOCAL_MODE=False,
        KAFKA_HOSTNAME="kafka:9092",
        KAFKA_INFERENCE_TOPIC="input-topic",
        KAFKA_INFERENCE_TOPIC_TIMEOUT=1000,
        KAFKA_RANKED_PROVIDERS_TOPIC="output-topic",
        KAFKA_RANKED_PROVIDERS_TOPIC_TIMEOUT=2000,
    )
    mock_create_consumer.side_effect = NoBrokersAvailable

    with pytest.raises(SystemExit) as e:
        connect_consumers_or_load_data(settings, logger)

    logger.error.assert_called_once_with(
        "Kakfa broker not found at given url: %s", "kafka:9092"
    )
    assert e.value.code == 1


@pytest.fixture
def sample_msg():
    return {
        "providers": [
            {
                MSG_PROVIDER_NAME: "TEST_PROVIDER",
                MSG_REGION_NAME: "test-region",
                MSG_INSTANCE_REQ: 2.0,
                MSG_INSTANCES_WITH_EXACT_FLAVORS: 1.0,
            }
        ],
        MSG_DEP_UUID: "1234-uuid-5678",
    }


@patch("src.inference.main.load_inference_settings")
@patch("src.inference.main.setup_mlflow")
@patch("src.inference.main.connect_consumers_or_load_data")
@patch("src.inference.main.send_message")
@patch("src.inference.main.create_message")
def test_run_single_provider_no_inference(
    mock_create_message,
    mock_send_message,
    mock_connect_consumers,
    mock_setup_mlflow,
    mock_load_settings,
    sample_msg,
):
    logger = MagicMock()

    settings = MagicMock()
    settings.LOCAL_MODE = True
    mock_load_settings.return_value = settings

    mock_setup_mlflow.return_value = MagicMock()

    mock_connect_consumers.return_value = ([sample_msg], [])

    mock_create_message.return_value = {"result": "ok"}

    run(logger=logger)

    provider_key = f"{sample_msg['providers'][0][MSG_PROVIDER_NAME]}-{sample_msg['providers'][0][MSG_REGION_NAME]}"
    expected_result = {
        provider_key: {
            K_CLASS: -1,
            K_REGR: -1,
            K_RES_EXACT: 0.5,
            **sample_msg["providers"][0],
        }
    }

    mock_create_message.assert_called_once()
    output_message = mock_create_message.call_args[1]
    assert output_message["sorted_results"] == expected_result
    mock_send_message.assert_called_once()


@pytest.fixture
def multi_provider_msg():
    return {
        MSG_DEP_UUID: "uuid-multi",
        "providers": [
            {
                MSG_PROVIDER_NAME: "PROV1",
                MSG_REGION_NAME: "reg1",
                MSG_INSTANCES_WITH_EXACT_FLAVORS: 1.0,
                MSG_INSTANCE_REQ: 1,
            },
            {
                MSG_PROVIDER_NAME: "PROV2",
                MSG_REGION_NAME: "reg2",
                MSG_INSTANCES_WITH_EXACT_FLAVORS: 0.5,
                MSG_INSTANCE_REQ: 2,
            },
        ],
    }


@pytest.fixture
def single_provider_msg():
    return {
        MSG_DEP_UUID: "uuid-single",
        "providers": [
            {
                MSG_PROVIDER_NAME: "PROV1",
                MSG_REGION_NAME: "reg1",
                "exact_flavors": 1.0,
                "n_instances_requ": 1,
            }
        ],
    }


@patch("src.inference.main.load_inference_settings")
@patch("src.inference.main.setup_mlflow")
@patch("src.inference.main.connect_consumers_or_load_data")
@patch("src.inference.main.send_message")
@patch("src.inference.main.create_message")
def test_local_mode_false_executes_blocks(
    mock_create_message,
    mock_send_message,
    mock_connect_consumers,
    mock_setup_mlflow,
    mock_load_settings,
    single_provider_msg,
):
    logger = MagicMock()

    settings = MagicMock()
    settings.LOCAL_MODE = False
    settings.LOCAL_DATASET = "dummy.csv"
    settings.LOCAL_DATASET_VERSION = "v1"
    settings.KAFKA_HOSTNAME = "kafka-host"
    settings.KAFKA_TRAINING_TOPIC = "topic"
    settings.KAFKA_TRAINING_TOPIC_PARTITION = 0
    settings.KAFKA_TRAINING_TOPIC_OFFSET = 0
    settings.KAFKA_TRAINING_TOPIC_TIMEOUT = 1000
    settings.TEMPLATE_COMPLEX_TYPES = []

    mock_load_settings.return_value = settings
    mock_setup_mlflow.return_value = MagicMock()

    mocked_message = MagicMock()
    mocked_message.value = single_provider_msg
    mock_connect_consumers.return_value = ([mocked_message], [])

    mock_create_message.return_value = {"result": "ok"}

    run(logger=logger)

    assert logger.debug.call_count > 0
    logger.debug.assert_any_call("Message: %s", ANY)


@patch("src.inference.main.load_inference_settings")
@patch("src.inference.main.setup_mlflow")
@patch("src.inference.main.connect_consumers_or_load_data")
@patch("src.inference.main.send_message")
@patch("src.inference.main.create_message")
def test_processed_dep_uuids_skips_message(
    mock_create_message,
    mock_send_message,
    mock_connect_consumers,
    mock_setup_mlflow,
    mock_load_settings,
    single_provider_msg,
):
    logger = MagicMock()
    settings = MagicMock()
    settings.LOCAL_MODE = True
    mock_load_settings.return_value = settings
    mock_setup_mlflow.return_value = MagicMock()

    mock_connect_consumers.return_value = (
        [single_provider_msg],  # input_consumer
        [{MSG_DEP_UUID: single_provider_msg[MSG_DEP_UUID]}],  # output_consumer
    )

    mock_create_message.return_value = {"result": "ok"}

    run(logger=logger)

    logger.info.assert_any_call("Already processed message. Skipping")


@patch("src.inference.main.load_inference_settings")
@patch("src.inference.main.setup_mlflow")
@patch("src.inference.main.connect_consumers_or_load_data")
@patch("src.inference.main.send_message")
@patch("src.inference.main.create_message")
def test_processed_dep_uuids_value_error_except(
    mock_create_message,
    mock_send_message,
    mock_connect_consumers,
    mock_setup_mlflow,
    mock_load_settings,
    single_provider_msg,
):
    logger = MagicMock()
    settings = MagicMock()
    settings.LOCAL_MODE = True
    mock_load_settings.return_value = settings
    mock_setup_mlflow.return_value = MagicMock()

    mock_connect_consumers.return_value = (
        [single_provider_msg],
        [{MSG_DEP_UUID: "uuid-not-present"}],
    )

    mock_create_message.return_value = {"result": "ok"}

    run(logger=logger)

    mock_create_message.assert_called_once()


@patch("src.inference.main.merge_and_sort_results")
@patch("src.inference.main.predict")
@patch("src.inference.main.pre_process_message")
@patch("src.inference.main.preprocessing")
@patch("src.inference.main.load_training_data")
@patch("src.inference.main.load_inference_settings")
@patch("src.inference.main.setup_mlflow")
@patch("src.inference.main.connect_consumers_or_load_data")
@patch("src.inference.main.send_message")
@patch("src.inference.main.create_message")
def test_multi_providers_path_executes(
    mock_create_message,
    mock_send_message,
    mock_connect_consumers,
    mock_setup_mlflow,
    mock_load_settings,
    mock_load_training_data,
    mock_preprocessing,
    mock_pre_process_message,
    mock_predict,
    mock_merge_and_sort_results,
    multi_provider_msg,
):
    logger = MagicMock()
    settings = MagicMock()
    settings.LOCAL_MODE = True
    settings.LOCAL_DATASET = "dummy.csv"
    settings.LOCAL_DATASET_VERSION = "v1"
    settings.KAFKA_HOSTNAME = "kafka-host"
    settings.KAFKA_TRAINING_TOPIC = "topic"
    settings.KAFKA_TRAINING_TOPIC_PARTITION = 0
    settings.KAFKA_TRAINING_TOPIC_OFFSET = 0
    settings.KAFKA_TRAINING_TOPIC_TIMEOUT = 1000
    settings.TEMPLATE_COMPLEX_TYPES = []

    mock_load_settings.return_value = settings
    mock_setup_mlflow.return_value = MagicMock()
    mock_connect_consumers.return_value = ([multi_provider_msg], [])
    mock_create_message.return_value = {"result": "ok"}

    mock_load_training_data.return_value = MagicMock()
    mock_preprocessing.return_value = MagicMock()
    mock_pre_process_message.return_value = MagicMock()
    mock_predict.return_value = (MagicMock(), MagicMock())
    mock_merge_and_sort_results.return_value = {"sorted": "results"}

    run(logger=logger)

    logger.info.assert_any_call("Select between multiple providers")


@pytest.mark.parametrize(
    "side_effect_exception, error_message",
    [
        (FileNotFoundError("File 'dummy.csv' not found"), "File 'dummy.csv' not found"),
        (
            NoBrokersAvailable("Kakfa broker not found at given url: kafka-host"),
            "Kakfa broker not found at given url: kafka-host",
        ),
        (AssertionError("Test assertion"), "Test assertion"),
        (ValueError("Test value error"), "Test value error"),
    ],
)
@patch("src.inference.main.load_inference_settings")
@patch("src.inference.main.setup_mlflow")
@patch("src.inference.main.connect_consumers_or_load_data")
@patch("src.inference.main.send_message")
@patch("src.inference.main.create_message")
def test_multi_provider_error_handling(
    mock_create_message,
    mock_send_message,
    mock_connect_consumers,
    mock_setup_mlflow,
    mock_load_settings,
    side_effect_exception,
    error_message,
    multi_provider_msg,
):
    logger = MagicMock()
    settings = MagicMock()
    settings.LOCAL_MODE = True
    settings.LOCAL_DATASET = "dummy.csv"
    settings.LOCAL_DATASET_VERSION = "v1"
    settings.KAFKA_HOSTNAME = "kafka-host"
    settings.KAFKA_TRAINING_TOPIC = "topic"
    settings.KAFKA_TRAINING_TOPIC_PARTITION = 0
    settings.KAFKA_TRAINING_TOPIC_OFFSET = 0
    settings.KAFKA_TRAINING_TOPIC_TIMEOUT = 1000
    settings.TEMPLATE_COMPLEX_TYPES = []

    mock_load_settings.return_value = settings
    mock_setup_mlflow.return_value = MagicMock()
    mock_connect_consumers.return_value = ([multi_provider_msg], [])
    mock_create_message.return_value = {"result": "ok"}

    with patch(
        "src.inference.main.load_training_data", side_effect=side_effect_exception
    ):
        run(logger=logger)

    def get_msg(call):
        try:
            return call.args[0] % call.args[1:]
        except Exception:
            return str(call.args)

    assert any(error_message in get_msg(call) for call in logger.error.call_args_list)


@patch("src.inference.main.connect_consumers_or_load_data")
@patch("src.inference.main.setup_mlflow")
@patch("src.inference.main.load_inference_settings")
def test_run_logs_info_if_no_providers(
    mock_load_inference_settings, mock_setup_mlflow, mock_connect_consumers_or_load_data
):
    mock_load_inference_settings.return_value.LOCAL_MODE = True
    test_message = {"providers": [], MSG_DEP_UUID: "test-uuid-1234"}
    input_consumer = [test_message]
    output_consumer = []
    mock_connect_consumers_or_load_data.return_value = (input_consumer, output_consumer)

    logger = MagicMock()

    run(logger=logger)

    logger.info.assert_any_call("No 'providers' available for this request")
    logger.error.assert_any_call("Inference process aborted")


@patch("src.inference.main.write_data_to_file")
@patch("src.inference.main.connect_consumers_or_load_data")
@patch("src.inference.main.setup_mlflow")
@patch("src.inference.main.load_inference_settings")
def test_run_sets_res_exact_to_zero_if_instance_req_is_zero(
    mock_load_inference_settings,
    mock_setup_mlflow,
    mock_connect_consumers_or_load_data,
    mock_write_data_to_file,
):
    mock_load_inference_settings.return_value.LOCAL_MODE = True

    test_message = {
        "providers": [
            {
                MSG_INSTANCE_REQ: 0,
                MSG_INSTANCES_WITH_EXACT_FLAVORS: 5,
                MSG_PROVIDER_NAME: "aws",
                MSG_REGION_NAME: "eu-west-1",
            }
        ],
        MSG_DEP_UUID: "uuid-test-5678",
    }

    input_consumer = [test_message]
    output_consumer = []
    mock_connect_consumers_or_load_data.return_value = (input_consumer, output_consumer)

    logger = MagicMock()

    run(logger=logger)

    logger.info.assert_any_call("Only one provider. No inference needed")
    mock_write_data_to_file.assert_called_once()
