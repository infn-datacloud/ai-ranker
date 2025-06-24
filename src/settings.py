from logging import Logger
from typing import Any

from pydantic import AnyHttpUrl, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsError
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
        default=None, description="Name of the file with the local dataset."
    )
    LOCAL_DATASET_VERSION: str = Field(
        default="1.1.0",
        description="Dataset's data were build following the target message version.",
    )

    KAFKA_HOSTNAME: str = Field(
        default="localhost:9092", description="Kafka server endpoint."
    )
    KAFKA_TRAINING_TOPIC: str = Field(
        default="training", description="Kafka default topic."
    )
    KAFKA_TRAINING_TOPIC_PARTITION: int | None = Field(
        default=None,
        ge=0,
        description="Training topic partition assigned to this consumer.",
    )
    KAFKA_TRAINING_TOPIC_OFFSET: int = Field(
        default=0, ge=0, description="Training topic read offset."
    )
    KAFKA_TRAINING_TOPIC_TIMEOUT: int = Field(
        default=1000,
        ge=0,
        description="Number of milliseconds to wait for a new message during iteration",
    )

    TEMPLATE_COMPLEX_TYPES: list = Field(
        default_factory=list, decription="List of complex template"
    )

    class Config:
        env_file = ".env"  # Set variables from env files
        env_file_encoding = "utf-8"
        extra = "ignore"


class TrainingSettings(CommonSettings):
    """Definition of environment variables related to the Training script."""

    CLASSIFICATION_MODELS: dict[str, ClassifierMixin] = Field(
        default={"RandomForestClassifier": {}},
        description="Pass a dict as a JSON string. The key is the model name. "
        "The value is a dict with the corresponding parameters",
    )
    REGRESSION_MODELS: dict[str, RegressorMixin] = Field(
        default={"RandomForestRegressor": {}},
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
    X_FEATURES: list = Field(
    default=[
        "cpu_diff",
        "ram_diff",
        "storage_diff",
        "instances_diff",
        "floatingips_diff",
        "gpu",
        "test_failure_perc_30d",
        "overbooking_ram",
        "avg_success_time",
        "avg_failure_time",
        "failure_percentage",
        "complexity",
    ],
    description="List of features to use as X"
)
    Y_CLASSIFICATION_FEATURES: list = Field(
        default=["status"],
        description="List of features to use as Y for classification"
    )
    Y_REGRESSION_FEATURES: list = Field(
        default=["deployment_time"],
        description="List of features to use as Y for regression"
    )

    SCALING_ENABLE: bool = Field(
        default=False, description="Perform the scaling of X features"
    )
    SCALER_FILE: str = Field(
        default="scaler.pkl", description="Default file where store the scaler"
    )

    @field_validator("CLASSIFICATION_MODELS", mode="before")
    @classmethod
    def parse_classification_models_params(
        cls, value: dict[str, dict]
    ) -> dict[str, ClassifierMixin]:
        """Verify the classification models"""
        return cls.__validate_models(value, "classifier", ClassifierMixin)

    @field_validator("REGRESSION_MODELS", mode="before")
    @classmethod
    def parse_regression_models_params(
        cls, value: dict[str, dict]
    ) -> dict[str, RegressorMixin]:
        """Verify the regression models"""
        return cls.__validate_models(value, "regressor", RegressorMixin)

    @classmethod
    def __validate_models(
        cls, value: dict[str, dict], model_type: str, model_class: type
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
            # Get the class of the model
            found_class = estimators.get(model_name, None)

            if found_class is None:
                raise ValueError(f"Model {model_name} not found")

            # Verify that the model belongs to the correct class
            if not issubclass(found_class, model_class):
                raise TypeError(f"The model {model_name} is not a {model_type} model")

            # Create the model object from the parameters
            model = found_class(**model_params)
            models_dict[model_name] = model

        return models_dict


class InferenceSettings(CommonSettings):
    """Definition of environment variables related to the Inference service"""

    CLASSIFICATION_MODEL_NAME: str = Field(
        default="RandomForestClassifier", description="Name of the classification model"
    )
    CLASSIFICATION_MODEL_VERSION: str = Field(
        default="latest", description="Version of classification model"
    )
    CLASSIFICATION_WEIGHT: float = Field(
        default=0.75, gt=0.0, lt=1.0, description="Classification weight"
    )

    REGRESSION_MODEL_NAME: str = Field(
        default="RandomForestRegressor", description="Name of the regressor model"
    )
    REGRESSION_MODEL_VERSION: str = Field(
        default="latest", description="Version of regression model"
    )

    LOCAL_IN_MESSAGES: str | None = Field(
        default=None, description="Name of the local input messages."
    )
    LOCAL_OUT_MESSAGES: str | None = Field(
        default=None, description="Name of the local outuput messages."
    )
    FILTER: bool = Field(default=False, description="Filter results under threshold")
    THRESHOLD: float = Field(default=0.7, description="Threshold to filter out score")
    EXACT_RESOURCES_PRECEDENCE: bool = Field(
        default=True,
        description="Sort providers based on how much the provider matches the "
        "requested resources.",
    )

    KAFKA_INFERENCE_TOPIC: str = Field(
        default="inference", description="Kafka default inference topic."
    )
    KAFKA_INFERENCE_TOPIC_PARTITION: int | None = Field(
        default=None,
        ge=0,
        description="Inference topic partition assigned to this consumer.",
    )
    KAFKA_INFERENCE_TOPIC_OFFSET: int = Field(
        default=0, ge=0, description="Inference topic read offset."
    )
    KAFKA_INFERENCE_TOPIC_TIMEOUT: int = Field(
        default=0,
        ge=0,
        description="Number of milliseconds to wait for a new message during iteration",
    )

    KAFKA_RANKED_PROVIDERS_TOPIC: str = Field(
        default="ranked_providers", description="Kafka default inference topic."
    )
    KAFKA_RANKED_PROVIDERS_TOPIC_PARTITION: int | None = Field(
        default=None,
        ge=0,
        description="Inference topic partition assigned to this consumer.",
    )
    KAFKA_RANKED_PROVIDERS_TOPIC_OFFSET: int = Field(
        default=0, ge=0, description="Inference topic read offset."
    )
    KAFKA_RANKED_PROVIDERS_TOPIC_TIMEOUT: int = Field(
        default=1000,
        ge=0,
        description="Number of milliseconds to wait when reading published messages",
    )


def load_training_settings(logger: Logger) -> TrainingSettings:
    """Function to load the training settings"""
    try:
        return TrainingSettings()
    except (ValidationError, SettingsError) as e:
        logger.error(e)
        exit(1)


def load_inference_settings(logger: Logger) -> InferenceSettings:
    """Function to load the inference settings"""
    try:
        return InferenceSettings()
    except (ValidationError, SettingsError) as e:
        logger.error(e)
        exit(1)


def load_mlflow_settings(logger: Logger) -> MLFlowSettings:
    """Function to load the mlflow settings"""
    try:
        return MLFlowSettings()
    except (ValidationError, SettingsError) as e:
        logger.error(e)
        exit(1)
