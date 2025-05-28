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

from src.training.main import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    get_feature_importance,
    remove_outliers,
    remove_outliers_from_dataframe,
)


def test_remove_outliers_from_dataframe():
    # Create a DataFrame with clear outliers
    data = {
        "A": [1, -10, 3, 4, 5, 100],  # Outlier at the end
        "B": [10, 20, 30, 40, 50, 200],  # Outlier at the end
        "C": [100, 200, 300, 400, 500, 600]  # No outliers in this column
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
    df = pd.DataFrame({
        "A": [10, 12, 14, 16, 18, 20],
        "B": [5, 7, 9, 11, 13, 15]
    })
    cleaned_df = remove_outliers_from_dataframe(df)
    pd.testing.assert_frame_equal(cleaned_df, df)


def test_remove_outliers_from_dataframe_constant_columns():
    # Columns with constant values should not generate outliers
    df = pd.DataFrame({
        "A": [5, 5, 5, 5, 5],
        "B": [10, 10, 10, 10, 10]
    })
    cleaned_df = remove_outliers_from_dataframe(df)
    pd.testing.assert_frame_equal(cleaned_df, df)


def test_remove_outliers_from_dataframe_single_extreme_value():
    # One extreme value in an otherwise uniform column
    df = pd.DataFrame({
        "A": [1, 1, 1, 1, 1000],
        "B": [2, 2, 2, 2, 2]
    })
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
    df = pd.DataFrame({
        "A": [1, 2, 3, None, 100],
        "B": [10, 20, 30, 40, 200]
    })
    df_clean = df.dropna()
    cleaned_df = remove_outliers_from_dataframe(df_clean)
    assert 100 not in cleaned_df["A"].values
    assert 200 not in cleaned_df["B"].values


def test_remove_outliers_from_dataframe_custom_params():
    # Custom IQR parameters to test different sensitivity to outliers
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 100]
    })

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
    df = pd.DataFrame({
        "A": [-100, -10, 0, 10, 20, 200],
        "B": [5, 5, 5, 5, 5, 5]
    })
    cleaned_df = remove_outliers_from_dataframe(df)
    assert -100 not in cleaned_df["A"].values
    assert 200 not in cleaned_df["A"].values
    assert all(cleaned_df["B"] == 5)


def test_remove_outliers_combined_basic():
    # Outliers in both X and Y should be removed together
    x = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 100],
        "B": [10, 20, 30, 40, 50, 200]
    })
    y = pd.DataFrame({
        "target": [0, 1, 0, 1, 0, 1]
    })
    x_clean, y_clean = remove_outliers(x, y)
    assert len(x_clean) == 5
    assert 100 not in x_clean["A"].values
    assert 200 not in x_clean["B"].values
    assert len(y_clean) == len(x_clean)


def test_remove_outliers_combined_no_outliers():
    # No outliers in either X or Y
    x = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50]
    })
    y = pd.DataFrame({
        "target": [1, 0, 1, 0, 1]
    })
    x_clean, y_clean = remove_outliers(x, y)
    pd.testing.assert_frame_equal(x, x_clean)
    pd.testing.assert_frame_equal(y, y_clean)


def test_remove_outliers_combined_outlier_in_y():
    # Outlier only in Y column
    x = pd.DataFrame({
        "A": [1, 2, 3, 4, 5]
    })
    y = pd.DataFrame({
        "target": [10, 20, 30, 40, 9999]
    })
    x_clean, y_clean = remove_outliers(x, y)
    assert 9999 not in y_clean["target"].values
    assert len(x_clean) == len(y_clean) == 4



def test_remove_outliers_combined_constant_columns():
    # Constant values in both X and Y, nothing to remove
    x = pd.DataFrame({
        "A": [5, 5, 5, 5]
    })
    y = pd.DataFrame({
        "target": [1, 1, 1, 1]
    })
    x_clean, y_clean = remove_outliers(x, y)
    pd.testing.assert_frame_equal(x, x_clean)
    pd.testing.assert_frame_equal(y, y_clean)


def test_remove_outliers_combined_column_names_preserved():
    # Ensure that column names are preserved after filtering
    x = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 100]
    })
    y = pd.DataFrame({
        "label": [0, 1, 0, 1, 1]
    })
    x_clean, y_clean = remove_outliers(x, y)
    assert list(x_clean.columns) == ["feature1"]
    assert list(y_clean.columns) == ["label"]


@pytest.fixture
def dummy_logger():
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.error = MagicMock()
    return logger

def test_feature_importance_with_feature_importances(dummy_logger):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = pd.DataFrame({
        "f1": [0, 1, 0, 1],
        "f2": [1, 0, 1, 0]
    })
    y = [0, 1, 0, 1]
    model.fit(X, y)

    result = get_feature_importance(model, X.columns, X, dummy_logger)
    assert "Feature" in result.columns
    assert "Importance" in result.columns
    assert len(result) == 2
    assert result.iloc[0]["Importance"] >= result.iloc[1]["Importance"]

def test_feature_importance_with_coef(dummy_logger):
    model = LogisticRegression()
    X = pd.DataFrame({
        "f1": [0, 1, 0, 1],
        "f2": [1, 0, 1, 0]
    })
    y = [0, 1, 0, 1]
    model.fit(X, y)

    result = get_feature_importance(model, X.columns, X, dummy_logger)
    assert "Feature" in result.columns
    assert "Coefficient" in result.columns
    assert len(result) == 2
    assert result.iloc[0]["Coefficient"] >= result.iloc[1]["Coefficient"]

@patch("src.training.main.shap.Explainer")
def test_feature_importance_with_shap(mock_shap, dummy_logger):
    # Crea un modello fake senza coef_ o feature_importances_
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

    # Istanza finta dell'explainer
    mock_explainer_instance = MagicMock()
    mock_explainer_instance.shap_values.side_effect = Exception("SHAP error")
    mock_kernel_explainer.return_value = mock_explainer_instance

    X = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.3, 0.4]})
    model = DummyModel()

    with pytest.raises(SystemExit):
        get_feature_importance(model, X.columns, X, dummy_logger)


def test_feature_importance_with_multiclass_coef(dummy_logger):
    model = LogisticRegression(multi_class="ovr")
    X = pd.DataFrame({
        "f1": [0, 1, 2, 3],
        "f2": [1, 0, 1, 0]
    })
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
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return pd.DataFrame(x_train), pd.DataFrame(x_test), pd.Series(y_train), pd.Series(y_test)

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
        "accuracy_train", "auc_train", "f1_train", "precision_train", "recall_train",
        "accuracy_test", "auc_test", "f1_test", "precision_test", "recall_test"
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

    # Verifica che tutte le metriche siano in [0, 1]
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

        # Verifica che siano state loggate tutte le metriche
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
            y_train_pred=pd.Series([0]),  # <--- solo 1 elemento!
            y_test_pred=pd.Series([1]),   # <--- solo 1 elemento!
            logger=logger,
        )

def test_imbalanced_labels(sample_dataset):
    x_train, x_test, y_train, y_test = sample_dataset

    # Sbilanciamento
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

    # F1, precision e recall possono essere bassi, ma devono esserci
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

def test_mlflow_logging_failure(monkeypatch, sample_dataset):
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

    monkeypatch.setattr("mlflow.log_metric", fake_log_metric)

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

    # R² = 1 in caso perfetto, MSE = MAE = RMSE = 0
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

    # R² deve essere < 1, MSE > 0
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

    # MSE dovrebbe essere elevato, R² < 0
    assert metrics["mse_train"] > 0
    assert metrics["r2_train"] < 0

def test_regression_inconsistent_lengths_should_fail():
    y = pd.Series([1, 2, 3])
    y_pred = pd.Series([1, 2])  # lunghezza diversa

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

    # filtra sia X che y per rimuovere righe con NaN in y
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
    y_train = pd.Series([1.0, 2.0, 3.0, 1000.0])  # outlier estremo
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
    assert metrics["mse_train"] > 2000  # Il MSE sarà grande per l'outlier
    assert metrics["r2_train"] < 1  # R2 inferiore a 1, ma non errore

def test_classification_metrics_inconsistent_lengths():
    x = pd.DataFrame([[0, 1], [1, 0]])
    y = pd.Series([0, 1])
    model = RandomForestClassifier().fit(x, y)
    logger = MagicMock()
    y_train_pred = pd.Series([0])  # dimensione errata

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