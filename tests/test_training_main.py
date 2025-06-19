from datetime import datetime
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.settings import TrainingSettings
from src.training.main import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    get_feature_importance,
    remove_outliers,
    remove_outliers_from_dataframe,
    split_and_clean_data,
    train_model,
)
from src.training.models import MetaData


@pytest.fixture(autouse=True, scope="session")
def setup_mlflow_experiment():
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("TestExperiment")


def test_remove_outliers_from_dataframe():
    # Create a DataFrame with clear outliers
    data = {
        "A": [1, -10, 3, 4, 5, 100],  # Outlier at the end
        "B": [10, 20, 30, 40, 50, 200],  # Outlier at the end
        "C": [100, 200, 300, 400, 500, 600],  # No outliers in this column
    }
    df = pd.DataFrame(data)
    cleaned_df = remove_outliers_from_dataframe(df)

    # Outliers should be removed
    assert len(cleaned_df) == 4
    assert 100 not in cleaned_df["A"].values
    assert 200 not in cleaned_df["B"].values
    assert 600 not in cleaned_df["B"].values


def test_remove_outliers_from_dataframe_no_outliers():
    # All values within a reasonable range, no outliers expected
    df = pd.DataFrame({"A": [10, 12, 14, 16, 18, 20], "B": [5, 7, 9, 11, 13, 15]})
    cleaned_df = remove_outliers_from_dataframe(df)
    pd.testing.assert_frame_equal(cleaned_df, df)


def test_remove_outliers_from_dataframe_constant_columns():
    # Columns with constant values should not generate outliers
    df = pd.DataFrame({"A": [5, 5, 5, 5, 5], "B": [10, 10, 10, 10, 10]})
    cleaned_df = remove_outliers_from_dataframe(df)
    pd.testing.assert_frame_equal(cleaned_df, df)


def test_remove_outliers_from_dataframe_single_extreme_value():
    # One extreme value in an otherwise uniform column
    df = pd.DataFrame({"A": [1, 1, 1, 1, 1000], "B": [2, 2, 2, 2, 2]})
    cleaned_df = remove_outliers_from_dataframe(df)
    assert len(cleaned_df) == 4
    assert 1000 not in cleaned_df["A"].values


def test_remove_outliers_from_dataframe_empty():
    # Edge case: empty DataFrame
    df = pd.DataFrame(columns=["A", "B"])
    cleaned_df = remove_outliers_from_dataframe(df)
    pd.testing.assert_frame_equal(cleaned_df, df)


def test_remove_outliers_from_dataframe_with_nan():
    # NaN rows should be handled correctly (we drop them before passing to the function)
    df = pd.DataFrame({"A": [1, 2, 3, None, 100], "B": [10, 20, 30, 40, 200]})
    df_clean = df.dropna()
    cleaned_df = remove_outliers_from_dataframe(df_clean)
    assert 100 not in cleaned_df["A"].values
    assert 200 not in cleaned_df["B"].values


def test_remove_outliers_from_dataframe_custom_params():
    # Custom IQR parameters to test different sensitivity to outliers
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 100]})

    df_clean_strict = remove_outliers_from_dataframe(df, q1=0.25, q3=0.75, k=1.0)
    assert 100 not in df_clean_strict["A"].values
    assert len(df_clean_strict) <= 5

    df_clean_very_strict = remove_outliers_from_dataframe(df, q1=0.4, q3=0.6, k=1.0)
    assert len(df_clean_very_strict) < len(df)

    df_clean_no_removal = remove_outliers_from_dataframe(df, q1=0.25, q3=0.75, k=100.0)
    assert len(df_clean_no_removal) == 6
    assert 100 in df_clean_no_removal["A"].values


def test_remove_outliers_from_dataframe_single_column():
    # Handle single-column DataFrame
    df = pd.DataFrame({"A": [1, 2, 3, 4, 1000]})
    cleaned_df = remove_outliers_from_dataframe(df)
    assert 1000 not in cleaned_df["A"].values


def test_remove_outliers_from_dataframe_negative_positive():
    # Mix of negative and positive values with outliers on both ends
    df = pd.DataFrame({"A": [-100, -10, 0, 10, 20, 200], "B": [5, 5, 5, 5, 5, 5]})
    cleaned_df = remove_outliers_from_dataframe(df)
    assert -100 not in cleaned_df["A"].values
    assert 200 not in cleaned_df["A"].values
    assert all(cleaned_df["B"] == 5)


def test_remove_outliers_combined_basic():
    # Outliers in both X and Y should be removed together
    x = pd.DataFrame({"A": [1, 2, 3, 4, 5, 100], "B": [10, 20, 30, 40, 50, 200]})
    y = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1]})
    x_clean, y_clean = remove_outliers(x, y)
    assert len(x_clean) == 5
    assert 100 not in x_clean["A"].values
    assert 200 not in x_clean["B"].values
    assert len(y_clean) == len(x_clean)


def test_remove_outliers_combined_no_outliers():
    # No outliers in either X or Y
    x = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
    y = pd.DataFrame({"target": [1, 0, 1, 0, 1]})
    x_clean, y_clean = remove_outliers(x, y)
    pd.testing.assert_frame_equal(x, x_clean)
    pd.testing.assert_frame_equal(y, y_clean)


def test_remove_outliers_combined_outlier_in_y():
    # Outlier only in Y column
    x = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    y = pd.DataFrame({"target": [10, 20, 30, 40, 9999]})
    x_clean, y_clean = remove_outliers(x, y)
    assert 9999 not in y_clean["target"].values
    assert len(x_clean) == len(y_clean) == 4


def test_remove_outliers_combined_constant_columns():
    # Constant values in both X and Y, nothing to remove
    x = pd.DataFrame({"A": [5, 5, 5, 5]})
    y = pd.DataFrame({"target": [1, 1, 1, 1]})
    x_clean, y_clean = remove_outliers(x, y)
    pd.testing.assert_frame_equal(x, x_clean)
    pd.testing.assert_frame_equal(y, y_clean)


def test_remove_outliers_combined_column_names_preserved():
    # Ensure that column names are preserved after filtering
    x = pd.DataFrame({"feature1": [1, 2, 3, 4, 100]})
    y = pd.DataFrame({"label": [0, 1, 0, 1, 1]})
    x_clean, y_clean = remove_outliers(x, y)
    assert list(x_clean.columns) == ["feature1"]
    assert list(y_clean.columns) == ["label"]


@pytest.fixture
def sample_data():
    x = pd.DataFrame(
        {
            "feature1": np.concatenate(
                [np.random.normal(50, 5, 95), [500, 600, 700, 800, 900]]
            ),
            "feature2": np.random.normal(0, 1, 100),
        }
    )
    y = pd.DataFrame({"target": np.random.randint(0, 2, size=100)})
    return x, y


def test_split_without_removing_outliers(sample_data):
    x, y = sample_data
    settings = TrainingSettings(
        TEST_SIZE=0.2,
        REMOVE_OUTLIERS=False,
        Q1_FACTOR=0.25,
        Q3_FACTOR=0.75,
        THRESHOLD_FACTOR=1.5,
    )
    x_train, x_test, y_train, y_test = split_and_clean_data(x=x, y=y, settings=settings)

    assert len(x_train) == 80
    assert len(x_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_split_with_removing_outliers(sample_data):
    x, y = sample_data
    settings = TrainingSettings(
        TEST_SIZE=0.2,
        REMOVE_OUTLIERS=True,
        Q1_FACTOR=0.25,
        Q3_FACTOR=0.75,
        THRESHOLD_FACTOR=1.5,
    )
    x_train, x_test, y_train, y_test = split_and_clean_data(x=x, y=y, settings=settings)

    assert len(x_train) < 80
    assert len(x_test) == 20
    assert len(y_train) == len(x_train)
    assert len(y_test) == 20


def test_test_set_unchanged_with_or_without_outliers(sample_data):
    x, y = sample_data

    settings_no_outliers = TrainingSettings(
        TEST_SIZE=0.2,
        REMOVE_OUTLIERS=False,
        Q1_FACTOR=0.25,
        Q3_FACTOR=0.75,
        THRESHOLD_FACTOR=1.5,
    )

    settings_with_outliers = TrainingSettings(
        TEST_SIZE=0.2,
        REMOVE_OUTLIERS=True,
        Q1_FACTOR=0.25,
        Q3_FACTOR=0.75,
        THRESHOLD_FACTOR=1.5,
    )

    _, x_test_no, _, y_test_no = split_and_clean_data(
        x=x, y=y, settings=settings_no_outliers
    )
    _, x_test_yes, _, y_test_yes = split_and_clean_data(
        x=x, y=y, settings=settings_with_outliers
    )

    pd.testing.assert_frame_equal(
        x_test_no.reset_index(drop=True), x_test_yes.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        y_test_no.reset_index(drop=True), y_test_yes.reset_index(drop=True)
    )


@pytest.mark.parametrize("test_size", [0.1, 0.3, 0.5])
def test_split_respects_test_size(sample_data, test_size):
    x, y = sample_data
    settings = TrainingSettings(
        TEST_SIZE=test_size,
        REMOVE_OUTLIERS=False,
        Q1_FACTOR=0.25,
        Q3_FACTOR=0.75,
        THRESHOLD_FACTOR=1.5,
    )
    x_train, x_test, y_train, y_test = split_and_clean_data(x=x, y=y, settings=settings)

    expected_test_len = int(len(x) * test_size)
    expected_train_len = len(x) - expected_test_len

    assert len(x_test) == expected_test_len
    assert len(x_train) == expected_train_len
    assert len(y_test) == expected_test_len
    assert len(y_train) == expected_train_len


def test_split_with_empty_dataset():
    x = pd.DataFrame(columns=["feature1", "feature2"])
    y = pd.DataFrame(columns=["target"])

    settings = TrainingSettings(
        TEST_SIZE=0.2,
        REMOVE_OUTLIERS=True,
        Q1_FACTOR=0.25,
        Q3_FACTOR=0.75,
        THRESHOLD_FACTOR=1.5,
    )

    with pytest.raises(ValueError):
        split_and_clean_data(x=x, y=y, settings=settings)


def test_split_with_single_sample():
    x = pd.DataFrame({"feature1": [1], "feature2": [2]})
    y = pd.DataFrame({"target": [0]})

    settings = TrainingSettings(
        TEST_SIZE=0.2,
        REMOVE_OUTLIERS=False,
        Q1_FACTOR=0.25,
        Q3_FACTOR=0.75,
        THRESHOLD_FACTOR=1.5,
    )

    with pytest.raises(ValueError):
        split_and_clean_data(x=x, y=y, settings=settings)


@pytest.fixture
def dummy_logger():
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.error = MagicMock()
    return logger


def test_feature_importance_with_feature_importances(dummy_logger):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 0, 1, 0]})
    y = [0, 1, 0, 1]
    model.fit(X, y)

    result = get_feature_importance(model, X.columns, X, dummy_logger)
    assert "Feature" in result.columns
    assert "Importance" in result.columns
    assert len(result) == 2
    assert result.iloc[0]["Importance"] >= result.iloc[1]["Importance"]


def test_feature_importance_with_coef(dummy_logger):
    model = LogisticRegression()
    X = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 0, 1, 0]})
    y = [0, 1, 0, 1]
    model.fit(X, y)

    result = get_feature_importance(model, X.columns, X, dummy_logger)
    assert "Feature" in result.columns
    assert "Coefficient" in result.columns
    assert len(result) == 2
    assert result.iloc[0]["Coefficient"] >= result.iloc[1]["Coefficient"]


@patch("src.training.main.shap.Explainer")
def test_feature_importance_with_shap(mock_shap, dummy_logger):
    class DummyModel(BaseEstimator):
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]] * len(X))

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array([[[0.1, 0.2], [0.2, 0.3]]])
    mock_shap.KernelExplainer.return_value = mock_explainer
    mock_shap.sample.side_effect = lambda x: x
    mock_shap.summary_plot = MagicMock()

    X = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.3, 0.4]})
    model = DummyModel()

    result = get_feature_importance(model, X.columns, X, dummy_logger)
    assert "Feature" in result.columns
    assert "Importance" in result.columns
    assert len(result) == 2
    assert result.iloc[0]["Importance"] >= 0


@patch("src.training.main.shap.KernelExplainer")
def test_feature_importance_shap_failure(mock_kernel_explainer, dummy_logger):
    class DummyModel(BaseEstimator):
        def predict_proba(self, X):
            return np.array([[0.5, 0.5]] * len(X))

    mock_explainer_instance = MagicMock()
    mock_explainer_instance.shap_values.side_effect = Exception("SHAP error")
    mock_kernel_explainer.return_value = mock_explainer_instance

    X = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.3, 0.4]})
    model = DummyModel()

    with pytest.raises(SystemExit):
        get_feature_importance(model, X.columns, X, dummy_logger)


def test_feature_importance_with_multiclass_coef(dummy_logger):
    model = LogisticRegression(multi_class="ovr")
    X = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
    y = [0, 1, 2, 1]
    model.fit(X, y)

    result = get_feature_importance(model, X.columns, X, dummy_logger)
    assert "Feature" in result.columns
    assert "Coefficient" in result.columns
    assert len(result) == 2


def test_feature_importance_mismatch_length_feature_importances(dummy_logger):
    class BadModel:
        feature_importances_ = [0.1]  # solo 1 valore

    X = pd.DataFrame({"f1": [0.1], "f2": [0.2]})
    model = BadModel()

    with pytest.raises(SystemExit):
        get_feature_importance(model, X.columns, X, dummy_logger)


def test_logger_called_in_feature_importance(dummy_logger):
    model = RandomForestClassifier(n_estimators=5)
    X = pd.DataFrame({"f1": [0, 1], "f2": [1, 0]})
    y = [0, 1]
    model.fit(X, y)

    get_feature_importance(model, X.columns, X, dummy_logger)
    dummy_logger.debug.assert_called()


@patch("src.training.main.shap.KernelExplainer")
def test_fallback_to_shap_when_no_attrs(mock_kernel_explainer, dummy_logger):
    class DummyModel(BaseEstimator):
        def predict_proba(self, X):
            return np.array([[0.5, 0.5]] * len(X))

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array([[[0.1, 0.2], [0.2, 0.3]]])
    mock_kernel_explainer.return_value = mock_explainer

    X = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.3, 0.4]})
    model = DummyModel()

    result = get_feature_importance(model, X.columns, X, dummy_logger)
    assert "Feature" in result.columns
    assert "Importance" in result.columns


def test_feature_importance_with_empty_data(dummy_logger):
    model = LogisticRegression()
    model.coef_ = np.array([[0.5, 0.5]])

    X = pd.DataFrame(columns=["f1", "f2"])

    with pytest.raises(SystemExit):
        get_feature_importance(model, X.columns, X, dummy_logger)


MODELS = [
    RandomForestClassifier(random_state=42),
    LogisticRegression(solver="liblinear", random_state=42),
    GradientBoostingClassifier(random_state=42),
]


@pytest.fixture(scope="module")
def sample_dataset():
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_classes=2,
        n_informative=10,
        random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return (
        pd.DataFrame(x_train),
        pd.DataFrame(x_test),
        pd.Series(y_train),
        pd.Series(y_test),
    )


@pytest.mark.parametrize("model", MODELS)
def test_metrics_for_different_models(model, sample_dataset):
    x_train, x_test, y_train, y_test = sample_dataset
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    logger = MagicMock()

    result = calculate_classification_metrics(
        model=model,
        x_train_scaled=x_train,
        x_test_scaled=x_test,
        y_train=y_train,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        logger=logger,
    )

    expected_keys = {
        "accuracy_train",
        "auc_train",
        "f1_train",
        "precision_train",
        "recall_train",
        "accuracy_test",
        "auc_test",
        "f1_test",
        "precision_test",
        "recall_test",
    }

    assert set(result.keys()) == expected_keys
    for value in result.values():
        assert isinstance(value, float)


def test_metrics_values_are_valid(sample_dataset):
    model = LogisticRegression(solver="liblinear", random_state=42)
    x_train, x_test, y_train, y_test = sample_dataset
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    logger = MagicMock()

    result = calculate_classification_metrics(
        model=model,
        x_train_scaled=x_train,
        x_test_scaled=x_test,
        y_train=y_train,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        logger=logger,
    )

    for k, v in result.items():
        assert 0.0 <= v <= 1.0, f"{k} fuori range: {v}"


def test_logging_called(sample_dataset):
    model = LogisticRegression(solver="liblinear", random_state=42)
    x_train, x_test, y_train, y_test = sample_dataset
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    logger = MagicMock()
    _ = calculate_classification_metrics(
        model=model,
        x_train_scaled=x_train,
        x_test_scaled=x_test,
        y_train=y_train,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        logger=logger,
    )

    logger.info.assert_called_once_with("Calculate classification metrics")
    assert logger.debug.call_count >= 3


def test_log_metrics_to_mlflow(sample_dataset):
    model = GradientBoostingClassifier(random_state=42)
    x_train, x_test, y_train, y_test = sample_dataset
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    logger = MagicMock()

    result = calculate_classification_metrics(
        model=model,
        x_train_scaled=x_train,
        x_test_scaled=x_test,
        y_train=y_train,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        logger=logger,
    )

    with mlflow.start_run():
        for k, v in result.items():
            mlflow.log_metric(k, v)

        assert all(isinstance(v, float) for v in result.values())


def test_missing_predictions_should_raise():
    x = pd.DataFrame([[0, 1], [1, 0]])
    y = pd.Series([0, 1])
    model = RandomForestClassifier().fit(x, y)
    logger = MagicMock()

    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        calculate_classification_metrics(
            model=model,
            x_train_scaled=x,
            x_test_scaled=x,
            y_train=y,
            y_test=y,
            y_train_pred=pd.Series([0]),
            y_test_pred=pd.Series([1]),
            logger=logger,
        )


def test_imbalanced_labels(sample_dataset):
    x_train, x_test, y_train, y_test = sample_dataset

    y_train = pd.Series(np.random.choice([0, 1], size=len(y_train), p=[0.95, 0.05]))
    y_test = pd.Series(np.random.choice([0, 1], size=len(y_test), p=[0.95, 0.05]))

    model = LogisticRegression(solver="liblinear").fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    logger = MagicMock()
    result = calculate_classification_metrics(
        model=model,
        x_train_scaled=x_train,
        x_test_scaled=x_test,
        y_train=y_train,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        logger=logger,
    )

    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_model_without_predict_proba_should_fail(sample_dataset):
    model = SVC(probability=False)
    x_train, x_test, y_train, y_test = sample_dataset
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    logger = MagicMock()
    with pytest.raises(SystemExit) as exc_info:
        calculate_classification_metrics(
            model=model,
            x_train_scaled=x_train,
            x_test_scaled=x_test,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=y_train_pred,
            y_test_pred=y_test_pred,
            logger=logger,
        )
    assert exc_info.value.code == 1


def test_mlflow_logging_failure(sample_dataset):
    x_train, x_test, y_train, y_test = sample_dataset
    model = LogisticRegression(solver="liblinear").fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    logger = MagicMock()
    metrics = calculate_classification_metrics(
        model=model,
        x_train_scaled=x_train,
        x_test_scaled=x_test,
        y_train=y_train,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        logger=logger,
    )

    def fake_log_metric(*args, **kwargs):
        raise RuntimeError("Simulated MLflow failure")

    with patch("mlflow.log_metric", new=fake_log_metric):
        with pytest.raises(RuntimeError, match="Simulated MLflow failure"):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)


def test_regression_metrics_basic():
    x = pd.DataFrame(np.arange(10).reshape(-1, 1))
    y = pd.Series(np.arange(10))

    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    logger = MagicMock()
    metrics = calculate_regression_metrics(
        y_train=y,
        y_test=y,
        y_train_pred=y_pred,
        y_test_pred=y_pred,
        logger=logger,
    )

    assert metrics["mse_train"] == pytest.approx(0.0)
    assert metrics["r2_train"] == pytest.approx(1.0)
    assert metrics["mae_train"] == pytest.approx(0.0)
    assert metrics["rmse_train"] == pytest.approx(0.0)


def test_regression_with_noise():
    rng = np.random.default_rng(seed=42)
    x = pd.DataFrame(np.arange(100).reshape(-1, 1))
    y = pd.Series(np.arange(100) + rng.normal(0, 5, 100))

    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    logger = MagicMock()
    metrics = calculate_regression_metrics(
        y_train=y,
        y_test=y,
        y_train_pred=y_pred,
        y_test_pred=y_pred,
        logger=logger,
    )

    assert 0 < metrics["mse_train"] < 30
    assert 0 < metrics["rmse_train"]
    assert 0 < metrics["mae_train"]
    assert 0 < metrics["r2_train"] <= 1


def test_regression_with_wrong_predictions():
    y = pd.Series([10, 20, 30])
    y_pred = pd.Series([0, 0, 0])

    logger = MagicMock()
    metrics = calculate_regression_metrics(
        y_train=y,
        y_test=y,
        y_train_pred=y_pred,
        y_test_pred=y_pred,
        logger=logger,
    )

    assert metrics["mse_train"] > 0
    assert metrics["r2_train"] < 0


def test_regression_inconsistent_lengths_should_fail():
    y = pd.Series([1, 2, 3])
    y_pred = pd.Series([1, 2])

    logger = MagicMock()
    with pytest.raises(ValueError):
        calculate_regression_metrics(
            y_train=y,
            y_test=y,
            y_train_pred=y_pred,
            y_test_pred=y_pred,
            logger=logger,
        )


def test_classification_metrics_with_nan():
    x = pd.DataFrame([[0, 1], [1, 0]])
    y = pd.Series([0, np.nan])

    mask = y.notna()
    x_clean = x[mask]
    y_clean = y[mask]

    model = RandomForestClassifier().fit(x_clean, y_clean)

    logger = MagicMock()
    y_pred = pd.Series([0, 1])

    with pytest.raises(ValueError):
        calculate_classification_metrics(
            model=model,
            x_train_scaled=x,
            x_test_scaled=x,
            y_train=y,
            y_test=y,
            y_train_pred=y_pred,
            y_test_pred=y_pred,
            logger=logger,
        )


def test_regression_metrics_with_outliers():
    y_train = pd.Series([1.0, 2.0, 3.0, 1000.0])
    y_train_pred = pd.Series([1.1, 2.1, 2.9, 900.0])
    y_test = pd.Series([1.5, 2.5, 3.5])
    y_test_pred = pd.Series([1.4, 2.4, 3.6])
    logger = MagicMock()

    metrics = calculate_regression_metrics(
        y_train=y_train,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        logger=logger,
    )
    assert metrics["mse_train"] > 2000
    assert metrics["r2_train"] < 1


def test_classification_metrics_inconsistent_lengths():
    x = pd.DataFrame([[0, 1], [1, 0]])
    y = pd.Series([0, 1])
    model = RandomForestClassifier().fit(x, y)
    logger = MagicMock()
    y_train_pred = pd.Series([0])

    with pytest.raises(ValueError):
        calculate_classification_metrics(
            model=model,
            x_train_scaled=x,
            x_test_scaled=x,
            y_train=y,
            y_test=y,
            y_train_pred=y_train_pred,
            y_test_pred=y_train_pred,
            logger=logger,
        )


def test_mlflow_logging_regression_metrics():
    y_train = pd.Series([1, 2, 3])
    y_test = pd.Series([4, 5, 6])
    y_train_pred = pd.Series([1.1, 2.1, 2.9])
    y_test_pred = pd.Series([3.9, 4.8, 6.2])
    logger = MagicMock()

    with mlflow.start_run() as run:
        metrics = calculate_regression_metrics(
            y_train=y_train,
            y_test=y_test,
            y_train_pred=y_train_pred,
            y_test_pred=y_test_pred,
            logger=logger,
        )
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    logged_metrics = client.get_run(run_id).data.metrics

    for k in metrics.keys():
        assert k in logged_metrics
        assert abs(metrics[k] - logged_metrics[k]) < 1e-6


def test_mlflow_logging_classification_metrics():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = RandomForestClassifier().fit(X, y)
    y_pred = model.predict(X)
    logger = MagicMock()

    with mlflow.start_run() as run:
        metrics = calculate_classification_metrics(
            model=model,
            x_train_scaled=pd.DataFrame(X),
            x_test_scaled=pd.DataFrame(X),
            y_train=pd.Series(y),
            y_test=pd.Series(y),
            y_train_pred=pd.Series(y_pred),
            y_test_pred=pd.Series(y_pred),
            logger=logger,
        )
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    logged_metrics = client.get_run(run_id).data.metrics

    for k in metrics.keys():
        assert k in logged_metrics
        assert abs(metrics[k] - logged_metrics[k]) < 1e-6


@pytest.fixture
def sample_data_complex():
    x = pd.DataFrame(
        {
            "feature1": np.concatenate(
                [np.random.normal(50, 5, 95), [500, 600, 700, 800, 900]]
            ),
            "feature2": np.random.normal(0, 1, 100),
        }
    )
    y = pd.DataFrame({"target": np.random.randint(0, 2, size=100)})

    train_len = int(0.8 * len(x))
    x_train = x.iloc[:train_len].reset_index(drop=True)
    x_test = x.iloc[train_len:].reset_index(drop=True)
    y_train = y.iloc[:train_len].reset_index(drop=True)
    y_test = y.iloc[train_len:].reset_index(drop=True)

    metadata = MetaData(
        features=list(x.columns),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    logger = MagicMock()

    return x_train, x_test, y_train, y_test, metadata, logger


@patch("src.training.main.get_feature_importance", return_value=pd.DataFrame())
@patch(
    "src.training.main.calculate_classification_metrics", return_value={"accuracy": 1.0}
)
@patch("src.training.main.log_on_mlflow")
def test_train_model_with_scaling_classification(
    mock_log, mock_metrics, mock_importance, sample_data_complex
):
    x_train, x_test, y_train, y_test, metadata, logger = sample_data_complex
    model = LogisticRegression()

    train_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        metadata=metadata,
        model_name="test_logreg",
        model=model,
        scaling_enable=True,
        scaler_file="scaler.pkl",
        logger=logger,
    )

    assert isinstance(model, LogisticRegression)
    mock_metrics.assert_called_once()
    mock_log.assert_called_once()
    mock_importance.assert_called_once()
    logger.info.assert_any_call("Model successfully trained")


@patch("src.training.main.get_feature_importance", return_value=pd.DataFrame())
@patch("src.training.main.calculate_regression_metrics", return_value={"mse": 0.1})
@patch("src.training.main.log_on_mlflow")
def test_train_model_without_scaling_regression(
    mock_log, mock_metrics, mock_importance, sample_data_complex
):
    x_train, x_test, y_train, y_test, metadata, logger = sample_data_complex
    model = LinearRegression()

    train_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        metadata=metadata,
        model_name="test_linreg",
        model=model,
        scaling_enable=False,
        scaler_file="scaler.pkl",
        logger=logger,
    )

    mock_metrics.assert_called_once()
    mock_log.assert_called_once()
    mock_importance.assert_called_once()
    logger.info.assert_any_call("Model successfully trained")


@patch("src.training.main.get_feature_importance", return_value=pd.DataFrame())
@patch("src.training.main.calculate_classification_metrics", return_value={})
@patch("src.training.main.log_on_mlflow")
def test_logger_usage(mock_log, mock_metrics, mock_importance, sample_data_complex):
    x_train, x_test, y_train, y_test, metadata, logger = sample_data_complex
    model = LogisticRegression()

    train_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        metadata=metadata,
        model_name="test_log",
        model=model,
        scaling_enable=True,
        scaler_file="scaler.pkl",
        logger=logger,
    )

    assert logger.info.call_count >= 3
    logger.info.assert_any_call("Metrics computed")
