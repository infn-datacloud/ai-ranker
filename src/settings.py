from logging import Logger

import mlflow
import mlflow.environment_variables
from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings
from sklearn.utils import all_estimators


class MLFlowSettings(BaseSettings):
    # Definition of environment variables with default values and description
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


class AIRankerTrainingSettings(BaseSettings):
    CLASSIFICATION_MODELS: dict[str, dict] = Field(
        description="Pass a dict as a JSON string. The key is the model name. "
        "The value is a dict with the corresponding parameters",
    )
    REGRESSION_MODELS: dict[str, dict] = Field(
        description="Pass a dict as a JSON string. The key is the model name. "
        "The value is a dict with the corresponding parameters",
    )

    LOCAL_MODE: bool = Field(
        default=False, description="Perform the training using local dataset"
    )
    LOCAL_DATASET: str = Field(default="", description="Name of the local dataset.")

    KFOLDS: int = Field(
        default=5, description="Number of folds for the KFold cross validation."
    )
    REMOVE_OUTLIERS: bool = Field(default=False, description="Remove outliers")
    FINAL_FEATURES: list = Field(
        description="List of final features in the processed Dataset",
    )

    TEMPLATE_COMPLEX_TYPES: list = Field(
        default_factory=list, decription="List of complex template"
    )

    KAFKA_URL: str = Field(
        default="localhost:9092", description="Kafka server endpoint."
    )
    KAFKA_TOPIC: str = Field(default="training", description="Kafka default topic.")

    @field_validator("CLASSIFICATION_MODELS", "REGRESSION_MODELS", mode="before")
    @classmethod
    def parse_models_params(cls, value: dict[str, dict]) -> dict[str, dict]:
        """Verify models and parameters.

        Model name (key) must belong to the scikit-learn library.
        Value must be a dict."""
        # Get all sklearn models (both classifiers and regressors)
        if isinstance(value, dict):
            estimators = [i[0] for i in all_estimators()]
            for k, v in value.items():
                assert k in estimators, f"Model '{k}' is not available in scikit-learn."
                assert isinstance(v, dict), f"Value of '{k}' is not a dictionary> {v}."
        return value

    class Config:
        env_file = ".env"  # Set variables from env files
        env_file_encoding = "utf-8"


class InferenceSettings(BaseSettings):
    CLASSIFICATION_MODEL_NAME: str = Field(
        default="RandomForestClassifier", description="Name of the classification model"
    )
    REGRESSION_MODEL_NAME: str = Field(
        default="RandomForestRegressor", description="Name of the regressor model"
    )
    CLASSIFICATION_MODEL_VERSION: str = Field(
        default="10", description="Version of classification model"
    )
    REGRESSION_MODEL_VERSION: str = Field(
        default="10", description="Version of regression model"
    )
    KAFKA_SERVER_URL: str = Field(default="localhost:9092", description="Kafka url")
    MIN_REGRESSION_TIME: int = Field(default=500, description="Minimum regression time")
    MAX_REGRESSION_TIME: int = Field(
        default=5000, description="Maximim regression time"
    )
    FILTER: bool = Field(default=False, description="Filter results undert threshold")
    THRESHOLD: float = Field(default=0.7, description="Threshold to filter out score")
    EXACT_FLAVOUR_PRECEDENCE: bool = Field(
        default=False,
        description="Sort providers putting them with exact flavour in front",
    )
    CLASSIFICATION_WEIGHT: float = Field(
        default=0.75, description="Classification weight"
    )
    TEMPLATE_COMPLEX_TYPES: list = Field(
        default=[
            "INDIGO IAM as a Service",
            "Elasticsearch and Kibana",
            "Kubernates cluster",
            "Spark + Jupyter cluster",
            "HTCondor mini",
            "HTCondor cluster",
            "Jupyter with persistence for Notebooks",
            "Jupyter + Matlab (with persistence for Notebooks)",
            "Computational enviroment for Machine Learning INFN (ML_INFN)",
            "Working Station for CYGNO experiment",
            "Sync&Share aaS",
        ],
        decription="List of complex template",
    )

    class Config:
        env_file = ".env"  # Set variables from env files
        env_file_encoding = "utf-8"


# Function to load the settings
def load_training_settings() -> AIRankerTrainingSettings:
    return AIRankerTrainingSettings()


# Function to load the settings
def load_inference_settings() -> InferenceSettings:
    return InferenceSettings()


# Function to load the settings
def load_mlflow_settings() -> MLFlowSettings:
    return MLFlowSettings()


def setup_mlflow(*, logger: Logger) -> None:
    """Set the mlflow server uri and experiment."""
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
