from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import all_estimators

from src import settings


def make_fake_validation_error(field: str, input_value: any, title: str = "Settings"):
    return ValidationError.from_exception_data(
        title=title,
        line_errors=[
            {
                "type": "string_type",
                "loc": (field,),
                "msg": "Input should be a valid string",
                "input": input_value,
            }
        ],
    )


def test_validate_classification_models_success():
    input_models = {"RandomForestClassifier": {"n_estimators": 10}}
    s = settings.TrainingSettings(
        CLASSIFICATION_MODELS=input_models,
        REGRESSION_MODELS={},
        X_FEATURES=[],
        Y_CLASSIFICATION_FEATURES=[],
        Y_REGRESSION_FEATURES=[],
    )
    assert "RandomForestClassifier" in s.CLASSIFICATION_MODELS
    assert s.CLASSIFICATION_MODELS["RandomForestClassifier"].n_estimators == 10


def test_invalid_model_name_raises_value_error():
    input_models = {"NonExistentModel": {}}
    with pytest.raises(ValueError, match="Model NonExistentModel not found"):
        settings.TrainingSettings(
            CLASSIFICATION_MODELS=input_models,
            REGRESSION_MODELS={},
            X_FEATURES=[],
            Y_CLASSIFICATION_FEATURES=[],
            Y_REGRESSION_FEATURES=[],
        )


@patch("src.settings.TrainingSettings", autospec=True)
def test_load_training_settings_success(mock_training_settings):
    logger = MagicMock()
    mock_training_settings.return_value = "loaded_training_settings"
    result = settings.load_training_settings(logger)
    assert result == "loaded_training_settings"
    logger.error.assert_not_called()


@patch("src.settings.TrainingSettings", autospec=True)
def test_load_training_settings_validation_error(mock_training_settings):
    logger = MagicMock()
    mock_training_settings.side_effect = make_fake_validation_error(
        "tracking_uri", None, "TrainingSettings"
    )
    with pytest.raises(SystemExit):
        settings.load_training_settings(logger)
    logger.error.assert_called_once_with(mock_training_settings.side_effect)


@patch("src.settings.InferenceSettings", autospec=True)
def test_load_inference_settings_success(mock_inference_settings):
    logger = MagicMock()
    mock_inference_settings.return_value = "loaded_inference_settings"
    result = settings.load_inference_settings(logger)
    assert result == "loaded_inference_settings"
    logger.error.assert_not_called()


@patch("src.settings.InferenceSettings", autospec=True)
def test_load_inference_settings_validation_error(mock_inference_settings):
    logger = MagicMock()
    mock_inference_settings.side_effect = make_fake_validation_error(
        "tracking_uri", None, "InferenceSettings"
    )
    with pytest.raises(SystemExit):
        settings.load_inference_settings(logger)
    logger.error.assert_called_once_with(mock_inference_settings.side_effect)


@patch("src.settings.MLFlowSettings", autospec=True)
def test_load_mlflow_settings_validation_error(mock_mlflow_settings):
    logger = MagicMock()
    mock_mlflow_settings.side_effect = make_fake_validation_error(
        "tracking_uri", None, "MLFlowSettings"
    )
    with pytest.raises(SystemExit):
        settings.load_mlflow_settings(logger)
    logger.error.assert_called_once_with(mock_mlflow_settings.side_effect)


@pytest.mark.parametrize(
    "type_filter, wrong_base_class, expected_type",
    [
        ("classifier", RegressorMixin, "classifier"),
        ("regressor", ClassifierMixin, "regressor"),
    ],
)
def test_validate_models_type_error(type_filter, wrong_base_class, expected_type):
    estimators = dict(all_estimators(type_filter=type_filter))
    model_name, _ = next(iter(estimators.items()))

    input_models = {model_name: {}}

    with pytest.raises(TypeError) as excinfo:
        settings.TrainingSettings._TrainingSettings__validate_models(
            input_models, expected_type, wrong_base_class
        )

    assert f"The model {model_name} is not a {expected_type} model" in str(
        excinfo.value
    )
