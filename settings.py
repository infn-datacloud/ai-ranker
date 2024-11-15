from pydantic import AnyHttpUrl, Field, field_validator, validator
from pydantic_settings import BaseSettings
from typing import Optional
import json
import ast

class MLflowSettings(BaseSettings):
    # Definition of environment variables with default values and description
    # Definizione delle variabili d'ambiente con valori di default e descrizioni
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        description="MLFlow tracking URI."
    )
    EXPERIMENT_NAME: str = Field(
        default="Default",
        description="Name of the MLFlow experiment."
    )
    MODEL_TYPE: str = Field(
        default="RandomForestClassifier",
        description="The type of ML model to train."
    )
    MODEL_PARAMS: str = Field(
        default="{}",
        description="Parameters to be passed to the model as a JSON string."
    )

    @field_validator("MODEL_PARAMS")
    def parse_model_params(cls, value):
        try:
            # convert the string into a dictionary
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error in MODEL_PARAMS parsing: {e}")
        
    class Config:
        env_file = ".env"  # Set variables from env files
        env_file_encoding = "utf-8"

# Function to load the settings
def load_mlflow_settings() -> MLflowSettings:
    return MLflowSettings()