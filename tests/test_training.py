from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.training.main import (
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
