from datetime import datetime

from pydantic import BaseModel, Field


class MetaData(BaseModel):
    start_time: datetime = Field(description="Datasert start time")
    end_time: datetime = Field(description="Datasert end time")
    features: list[str] = Field(description="Final features")
    features_number: int = Field(description="Number of features")
    remove_outliers: bool = Field(default=False, description="Flag to remove outliers")
