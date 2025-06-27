# ------------------------------
# MetaData Tests
# ------------------------------
from datetime import datetime

import pytest

from src.training.models import ClassificationMetrics, MetaData, RegressionMetrics


def get_basic_metadata_dict():
    return {
        "start_time": datetime(2024, 1, 1),
        "end_time": datetime(2024, 1, 2),
        "features": ["a", "b", "c"],
    }


def test_metadata_valid_construction_defaults():
    data = get_basic_metadata_dict()
    metadata = MetaData(**data)
    assert metadata.features_number == 3
    assert metadata.remove_outliers is False
    assert metadata.scaling is False
    assert metadata.scaler_file is None


def test_metadata_with_all_fields():
    data = get_basic_metadata_dict() | {
        "remove_outliers": True,
        "scaling": True,
        "scaler_file": "scaler.pkl",
    }
    metadata = MetaData(**data)
    assert metadata.remove_outliers is True
    assert metadata.scaler_file == "scaler.pkl"


def test_metadata_override_features_number():
    data = get_basic_metadata_dict() | {"features_number": 99}
    metadata = MetaData(**data)
    assert metadata.features_number == 99


def test_metadata_missing_required_fields():
    with pytest.raises(Exception):
        MetaData(end_time=datetime(2024, 1, 2), features=["a", "b"])  # manca start_time


def test_metadata_invalid_features_type():
    with pytest.raises(Exception):
        MetaData(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            features="not-a-list",
        )


def test_metadata_model_dump():
    metadata = MetaData(**get_basic_metadata_dict())
    dump = metadata.model_dump()
    assert dump["features_number"] == 3
    assert isinstance(dump["start_time"], datetime)


def test_metadata_json_serialization():
    metadata = MetaData(**get_basic_metadata_dict())
    json_data = metadata.model_dump()
    assert json_data["features_number"] == 3


def test_metadata_equality():
    d = get_basic_metadata_dict()
    assert MetaData(**d) == MetaData(**d)


# ------------------------------
# ClassificationMetrics Tests
# ------------------------------


def test_classification_metrics_valid():
    m = ClassificationMetrics(
        accuracy=0.9, auc=0.88, f1=0.87, precision=0.91, recall=0.85
    )
    assert m.f1 == 0.87


def test_classification_metrics_invalid_type():
    with pytest.raises(Exception):
        ClassificationMetrics(
            accuracy="high", auc=0.88, f1=0.87, precision=0.91, recall=0.85
        )


def test_classification_metrics_missing_field():
    with pytest.raises(Exception):
        ClassificationMetrics(
            accuracy=0.9,
            auc=0.88,
            f1=0.87,
            precision=0.91,
        )


def test_classification_metrics_serialization():
    m = ClassificationMetrics(
        accuracy=0.9, auc=0.88, f1=0.87, precision=0.91, recall=0.85
    )
    d = m.model_dump()
    assert d["accuracy"] == 0.9
    assert "recall" in d


def test_classification_metrics_json():
    m = ClassificationMetrics(
        accuracy=0.9, auc=0.88, f1=0.87, precision=0.91, recall=0.85
    )
    json_data = m.model_dump()
    assert json_data["accuracy"] == 0.9


# ------------------------------
# RegressionMetrics Tests
# ------------------------------


def test_regression_metrics_valid():
    m = RegressionMetrics(mse=12.0, rmse=3.46, mae=2.5, r2=0.88)
    assert m.r2 == 0.88


def test_regression_metrics_invalid_type():
    with pytest.raises(Exception):
        RegressionMetrics(mse=12.0, rmse="bad", mae=2.5, r2=0.88)


def test_regression_metrics_missing_field():
    with pytest.raises(Exception):
        RegressionMetrics(
            mse=12.0,
            rmse=3.46,
            r2=0.88,
        )


def test_regression_metrics_serialization():
    m = RegressionMetrics(mse=10.0, rmse=3.16, mae=2.0, r2=0.87)
    d = m.model_dump()
    assert d["rmse"] == pytest.approx(3.16, rel=1e-2)


def test_regression_metrics_json():
    m = RegressionMetrics(mse=10.0, rmse=3.16, mae=2.0, r2=0.87)
    json_data = m.model_dump()
    assert json_data["rmse"] == 3.16
