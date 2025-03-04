from datetime import datetime

from pydantic import BaseModel, Field


class MetaData(BaseModel):
    """Metadata stored in the MLFlow model registry."""

    start_time: datetime = Field(description="Datasert start time")
    end_time: datetime = Field(description="Datasert end time")
    features: list[str] = Field(description="Final features")
    features_number: int = Field(description="Number of features")
    remove_outliers: bool = Field(default=False, description="Flag to remove outliers")


class ClassificationMetrics(BaseModel):
    """Metrics stored in the MLFlow model registry."""

    accuracy: float = Field(description="Accuracy score on dataset")
    auc: float = Field(descrption="ROC AUC score on dataset")
    f1: float = Field(description="F1 score on dataset")
    precision: float = Field(description="Precision score on dataset")
    recall: float = Field(description="Recall score on dataset")


class RegressionMetrics(BaseModel):
    """Metrics stored in the MLFlow model registry."""

    mse: float = Field(description="Mean Square Error")
    rmse: float = Field(descrption="Root Mean Square Error")
    mae: float = Field(description="Mean Absolute Error")
    r2: float = Field(description="R2 score")
