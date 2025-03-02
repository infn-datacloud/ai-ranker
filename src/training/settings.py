import json

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class AIRankerTrainingSettings(BaseSettings):
    # Definition of environment variables with default values and description
    # Definizione delle variabili d'ambiente con valori di default e descrizioni
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000", description="MLFlow tracking URI."
    )
    EXPERIMENT_NAME: str = Field(
        default="Default", description="Name of the MLFlow experiment."
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


    @field_validator("CLASSIFICATION_MODELS", "REGRESSION_MODELS", mode="before")
    @classmethod
    def parse_models_params(cls, value: dict[str, dict]) -> dict[str, dict]:
        """Verify that the content is a dictionary."""
        if isinstance(value, dict):
            for k, v in value.items():
                assert isinstance(v, dict), f"Parameter {k} is not a dictionary."
        return value

    class Config:
        env_file = ".env"  # Set variables from env files
        env_file_encoding = "utf-8"


# Function to load the settings
def load_airankertraining_settings() -> AIRankerTrainingSettings:
    return AIRankerTrainingSettings()
