from logging import Logger
from typing import Any

import mlflow
import mlflow.environment_variables
from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import all_estimators


class MLFlowSettings(BaseSettings):
    """Definition of environment variables related to the MLFlow configuration."""

    MLFLOW_TRACKING_URI: AnyHttpUrl = Field(
        default="http://localhost:5000", description="MLFlow tracking URI."
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="Default", description="Name of the MLFlow experiment."
    )
    MLFLOW_HTTP_REQUEST_TIMEOUT: int = Field(
        default=20, decription="Timeout in seconds for MLflow HTTP requests"
    )
    MLFLOW_HTTP_REQUEST_MAX_RETRIES: int = Field(
        default=5,
        decription="Specifies the maximum number of retries with exponential backoff "
        "for MLflow HTTP requests ",
    )
    MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR: int = Field(
        default=2,
        decription="Specifies the backoff increase factor between MLflow HTTP request",
    )
    MLFLOW_HTTP_REQUEST_BACKOFF_JITTER: float = Field(
        default=1.0,
        decription="Specifies the backoff jitter between MLflow HTTP request failures",
    )

    class Config:
        env_file = ".env"  # Set variables from env files
        env_file_encoding = "utf-8"
        extra = "ignore"


class CommonSettings(BaseSettings):
    """Common settings"""

    LOCAL_MODE: bool = Field(
        default=False, description="Perform the training using local dataset"
    )
    LOCAL_DATASET: str | None = Field(
        default=None, description="Name of the local dataset."
    )

    KAFKA_URL: str = Field(
        default="localhost:9092", description="Kafka server endpoint."
    )
    KAFKA_TRAINING_TOPIC: str = Field(
        default="inference", description="Kafka default topic."
    )

    TEMPLATE_COMPLEX_TYPES: list = Field(
        default_factory=list, decription="List of complex template"
    )

    SCALER_FILE: str = Field(
        default="scaler.pkl", description="Default file where store the scaler"
    )

    SCALING_ENABLE: bool = Field(
        default=False, description="Perform the scaling of X features"
    )

    class Config:
        env_file = ".env"  # Set variables from env files
        env_file_encoding = "utf-8"
        extra = "ignore"


class TrainingSettings(CommonSettings):
    """Definition of environment variables related to the Training script."""

    CLASSIFICATION_MODELS: dict[str, dict] = Field(
        description="Pass a dict as a JSON string. The key is the model name. "
        "The value is a dict with the corresponding parameters",
    )
    REGRESSION_MODELS: dict[str, dict] = Field(
        description="Pass a dict as a JSON string. The key is the model name. "
        "The value is a dict with the corresponding parameters",
    )

    KFOLDS: int = Field(
        default=5, description="Number of folds for the KFold cross validation."
    )
    REMOVE_OUTLIERS: bool = Field(default=False, description="Remove outliers")
    TEST_SIZE: float = Field(
        default=0.2,
        lt=1.0,
        gt=0.0,
        description="Test size with respect to the whole dataset. Range (0, 1).",
    )
    Q1_FACTOR: float = Field(default=0.25, description="First quantile")
    Q3_FACTOR: float = Field(default=0.75, description="Third quantile")
    THRESHOLD_FACTOR: float = Field(
        default=1.5, description="Multiplication factor for outlier threshold"
    )
    FINAL_FEATURES: list = Field(
        description="List of final features in the processed Dataset",
    )

    @field_validator("CLASSIFICATION_MODELS", mode="after")
    @classmethod
    def parse_classification_models_params(
        cls, value: dict[str, dict], logger: Logger
    ) -> dict[str, Any]:
        """Verify the classification models"""
        return validate_models(value, "classifier", ClassifierMixin, logger)

    @field_validator("REGRESSION_MODELS", mode="after")
    @classmethod
    def parse_regression_models_params(
        cls, value: dict[str, dict], logger: Logger
    ) -> dict[str, Any]:
        """Verify the regression models"""
        return validate_models(value, "regressor", RegressorMixin, logger)


class InferenceSettings(CommonSettings):
    """Definition of environment variables related to the Inference service"""

    CLASSIFICATION_MODEL_NAME: str = Field(
        default="RandomForestClassifier", description="Name of the classification model"
    )
    CLASSIFICATION_MODEL_VERSION: str = Field(
        default="10", description="Version of classification model"
    )
    CLASSIFICATION_WEIGHT: float = Field(
        default=0.75, description="Classification weight"
    )

    REGRESSION_MODEL_NAME: str = Field(
        default="RandomForestRegressor", description="Name of the regressor model"
    )
    REGRESSION_MODEL_VERSION: str = Field(
        default="10", description="Version of regression model"
    )
    REGRESSION_MIN_TIME: int = Field(default=500, description="Minimum regression time")
    REGRESSION_MAX_TIME: int = Field(
        default=5000, description="Maximim regression time"
    )

    FILTER: bool = Field(default=False, description="Filter results undert threshold")
    THRESHOLD: float = Field(default=0.7, description="Threshold to filter out score")
    EXACT_FLAVOUR_PRECEDENCE: bool = Field(
        default=False,
        description="Sort providers putting them with exact flavour in front",
    )


def load_training_settings(logger: Logger) -> TrainingSettings:
    """Function to load the training settings"""
    return TrainingSettings(logger)


def load_inference_settings() -> InferenceSettings:
    """Function to load the inference settings"""
    return InferenceSettings()


def load_mlflow_settings() -> MLFlowSettings:
    """Function to load the mlflow settings"""
    return MLFlowSettings()

def setup_mlflow(*, logger: Logger) -> None:
    """Function to set up the mlflow settings"""
    logger.info("Setting up MLFlow service communication")
    settings = load_mlflow_settings()
    try:
        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_TIMEOUT.set(
            settings.MLFLOW_HTTP_REQUEST_TIMEOUT
        )
        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_MAX_RETRIES.set(
            settings.MLFLOW_HTTP_REQUEST_MAX_RETRIES
        )
        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR.set(
            settings.MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR
        )
        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_BACKOFF_JITTER.set(
            settings.MLFLOW_HTTP_REQUEST_BACKOFF_JITTER
        )

        mlflow.set_tracking_uri(str(settings.MLFLOW_TRACKING_URI))
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    except mlflow.exceptions.MlflowException as e:
        logger.error(e.message)
        exit(1)


def validate_models(
    value: dict[str, dict], model_type: str, model_class: type, logger: Logger
) -> dict[str, Any]:
    """Function to validate classifiers and regressors

    Args:
        value (dict): Dictionary with models and parameters
        model_type (str): Model type ("classifier" or "regressor").
        model_class (type): Class type (ClassifierMixin o RegressorMixin).

    Returns:
        dict: dictionary where values are the model objects
    """
    models_dict = {}
    estimators = dict(all_estimators(type_filter=model_type))

    for model_name, model_params in value.items():
        try:
            # Get the class of the model
            model_class = estimators.get(model_name, None)

            if model_class is None:
                raise ValueError(f"Model {model_name} not found")

            # Verify that the model belongs to the correct class
            if not issubclass(model_class, model_class):
                raise TypeError(f"The model {model_name} is not a {model_type} model")

            # Create the model object from the parameters
            model = model_class(**model_params)
            models_dict[model_name] = model

        except (TypeError, ValueError) as e:
            logger.error(
                e.message
            )  # FIX: 'pydantic_core._pydantic_core.ValidationInfo' object has no attribute 'error'
            exit(1)
    return models_dict
