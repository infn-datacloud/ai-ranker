from pydantic import Field
from pydantic_settings import BaseSettings


class MLflowSettings(BaseSettings):
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000", description="MLFlow tracking URI"
    )
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
def load_mlflow_settings() -> MLflowSettings:
    return MLflowSettings()
