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
    CLASSIFICATION_MODELS: list = Field(
        default=["RandomForestClassifier"],
        description="The list of ML models of type classification to compare.",
    )
    CLASSIFICATION_MODELS_PARAMS: str = Field(
        default="",
        description="Parameters to be passed to the classification models as a JSON string.",
    )
    REGRESSION_MODELS: list = Field(
        default=["RandomForestRegressor"],
        description="The list of ML models of type regression to compare.",
    )
    REGRESSION_MODELS_PARAMS: str = Field(
        default="",
        description="Parameters to be passed to the regression models as a JSON string.",
    )
    KFOLDS: int = Field(
        default=5, description="Number of folds for the KFold cross validation."
    )
    REMOVE_OUTLIERS: bool = Field(default=False, description="Remove outliers")
    LOCAL_MODE: bool = Field(
        default=False, description="Perform the training using local dataset"
    )
    LOCAL_DATASET: str = Field(default="", description="Name of the local dataset.")
    FINAL_FEATURES: list = Field(
        default=[], description="List of final features in the processed Dataset"
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

    @field_validator(
        "CLASSIFICATION_MODELS_PARAMS", "REGRESSION_MODELS_PARAMS", mode="before"
    )
    def parse_models_params(cls, value):
        # Verify that the content is a dictionary
        if isinstance(value, dict):
            try:
                parsed_value = json.loads(value)
                # Check if MODELS_PARAMS is a dictionary of dictionaries
                if not all(isinstance(v, dict) for v in parsed_value.values()):
                    raise ValueError(
                        "CLASSIFICATION_MODELS_PARAMS should be a dictionary of dictionaries."
                    )
                return parsed_value
            except json.JSONDecodeError as e:
                raise ValueError(f"Error in CLASSIFICATION_MODELS_PARAMS parsing: {e}")
        return value

    class Config:
        env_file = ".env"  # Set variables from env files
        env_file_encoding = "utf-8"


# Function to load the settings
def load_airankertraining_settings() -> AIRankerTrainingSettings:
    return AIRankerTrainingSettings()
