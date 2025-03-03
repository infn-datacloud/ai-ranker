from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings
from sklearn.utils import all_estimators


class AIRankerTrainingSettings(BaseSettings):
    # Definition of environment variables with default values and description
    # Definizione delle variabili d'ambiente con valori di default e descrizioni
    MLFLOW_TRACKING_URI: AnyHttpUrl = Field(
        default="http://localhost:5000", description="MLFlow tracking URI."
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="Default", description="Name of the MLFlow experiment."
    )
    MLFLOW_HTTP_REQUEST_TIMEOUT: int = Field(
        default=20, decription="Timeout in seconds for MLflow HTTP requests"
    )

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

    KAFKA_URL: AnyHttpUrl = Field(
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


# Function to load the settings
def load_airankertraining_settings() -> AIRankerTrainingSettings:
    return AIRankerTrainingSettings()
