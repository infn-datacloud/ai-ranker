import base64
import pickle
from logging import Logger
from tempfile import NamedTemporaryFile

import mlflow
import mlflow.environment_variables
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler

from processing import DF_TIMESTAMP, load_dataset, preprocessing
from settings import TrainingSettings, load_training_settings, setup_mlflow
from training.models import ClassificationMetrics, MetaData, RegressionMetrics

STATUS_COL = -2
DEP_TIME_COL = -1


def remove_outliers_from_dataframe(
    df: pd.DataFrame, *, q1: float = 0.25, q3: float = 0.75, k: float = 1.5
) -> pd.DataFrame:
    """Compute quantiles for all columns, define limits and remove outliers."""
    q1_series = df.quantile(q1)
    q3_series = df.quantile(q3)
    iqr = q3_series - q1_series
    lower_bound = q1_series - k * iqr
    upper_bound = q3_series + k * iqr
    return df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]


def remove_outliers(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    q1: float = 0.25,
    q3: float = 0.75,
    k: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concat X and Y, remove outliers and split X from Y."""
    combined = pd.concat([x, y], axis=1)
    filtered = remove_outliers_from_dataframe(combined, q1=q1, q3=q3, k=k)
    return filtered.iloc[:, :-1], filtered.iloc[:, -1]


def get_feature_importance(
    model: BaseEstimator,
    columns: pd.Index,
    x_train_scaled: pd.DataFrame,
    logger: Logger,
) -> pd.DataFrame:
    """Calculate feature importance in different ways depending on the model"""
    # Case 1: Models with attribute `feature_importances_`
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame(
            {"Feature": columns, "Importance": feature_importances}
        ).sort_values(by="Importance", ascending=False)
        logger.debug("Feature Importance:")
        logger.debug(feature_importance_df)
        return feature_importance_df

    # Case 2: Models like Lasso (coefficient)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        feature_importance_df = pd.DataFrame(
            {"Feature": columns, "Coefficient": coef}
        ).sort_values(by="Coefficient", ascending=False)
        logger.debug("Feature importance:")
        logger.debug(feature_importance_df)
        return feature_importance_df

    # Case 3: Models without `feature_importances_` or `coef_`, use SHAP
    else:
        try:
            # If the model is not a tree use KernelExplainer
            background_data_summarized = shap.sample(x_train_scaled[:50])
            explainer = shap.KernelExplainer(
                model.predict_proba, background_data_summarized
            )
            shap_values = explainer.shap_values(x_train_scaled)
            shap.summary_plot(shap_values, x_train_scaled)
            # Compute feature importance
            shap_importance = np.mean(np.abs(shap_values), axis=0)[:, 0]
            feature_importance_df = pd.DataFrame(
                {"Feature": columns, "Importance": shap_importance}
            ).sort_values(by="Importance", ascending=False)
            logger.debug("Feature importance:")
            logger.debug(feature_importance_df)
            return feature_importance_df
        except Exception as e:
            print(f"Error in using SHAP: {e}")
            return None


def calculate_classification_metrics(
    model: ClassifierMixin,
    *,
    x_train_scaled: pd.DataFrame,
    x_test_scaled: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    y_train_pred: pd.DataFrame,
    y_test_pred: pd.DataFrame,
    logger: Logger,
) -> dict[str, float]:
    """Compute Metrics on a Classification Model."""
    train_metrics = ClassificationMetrics(
        accuracy=accuracy_score(y_train, y_train_pred),
        auc=roc_auc_score(y_train, model.predict_proba(x_train_scaled)[:, 1]),
        f1=f1_score(y_train, y_train_pred, average="binary"),
        precision=precision_score(y_train, y_train_pred, average="binary"),
        recall=recall_score(y_train, y_train_pred, average="binary"),
    )
    logger.debug("Model metrics on the training dataset: %s", train_metrics)
    test_metrics = ClassificationMetrics(
        accuracy=accuracy_score(y_test, y_test_pred),
        auc=roc_auc_score(y_test, model.predict_proba(x_test_scaled)[:, 1]),
        f1=f1_score(y_test, y_test_pred, average="binary"),
        precision=precision_score(y_test, y_test_pred, average="binary"),
        recall=recall_score(y_test, y_test_pred, average="binary"),
    )
    logger.debug("Model metrics on the test dataset: %s", test_metrics)

    logger.debug("Confusion matrix:")
    logger.debug(confusion_matrix(y_test, y_test_pred))
    logger.debug("Classification report:")
    logger.debug(classification_report(y_test, y_test_pred))

    d1 = {f"{k}_train": v for k, v in train_metrics.model_dump().items()}
    d2 = {f"{k}_test": v for k, v in test_metrics.model_dump().items()}
    return {**d1, **d2}


def calculate_regression_metrics(
    *,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    y_train_pred: pd.DataFrame,
    y_test_pred: pd.DataFrame,
    logger: Logger,
) -> dict[str, float]:
    """Compute Metrics on a Regression Model."""

    train_metrics = RegressionMetrics(
        mse=mean_squared_error(y_train, y_train_pred),
        rmse=np.sqrt(mean_squared_error(y_train, y_train_pred)),
        mae=mean_absolute_error(y_train, y_train_pred),
        r2=r2_score(y_train, y_train_pred),
    )
    logger.debug("Train dataset metrics: %s", train_metrics)
    test_metrics = RegressionMetrics(
        mse=mean_squared_error(y_test, y_test_pred),
        rmse=np.sqrt(mean_squared_error(y_test, y_test_pred)),
        mae=mean_absolute_error(y_test, y_test_pred),
        r2=r2_score(y_test, y_test_pred),
    )
    logger.debug("Test dataset metrics: %s", test_metrics)

    d1 = {f"{k}_train": v for k, v in train_metrics.model_dump().items()}
    d2 = {f"{k}_test": v for k, v in test_metrics.model_dump().items()}
    return {**d1, **d2}


def train_model(
    *,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    metadata: MetaData,
    model_name: str,
    model: BaseEstimator,
    scaling_enable: bool,
    scaler_file: str,
    logger: Logger,
) -> None:
    """Function to train a generic sklearn ML model"""

    # Scale x_train if scaling is enabled
    if scaling_enable:
        scaler = RobustScaler()
        x_train_scaled = pd.DataFrame(
            scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index
        )
        scaler_bytes = pickle.dumps(scaler)
        x_test_scaled = pd.DataFrame(
            scaler.transform(x_test), columns=x_test.columns, index=x_test.index
        )
    else:
        x_train_scaled = x_train
        x_test_scaled = x_test
        scaler_bytes = None

    model_params = model.get_params()
    # Train the model
    logger.info("Training model '%s' with params: %s", model_name, model_params)
    model.fit(x_train_scaled, y_train.values.ravel())

    # Get feature importance
    feature_importance_df = get_feature_importance(
        model, x_train.columns, x_train_scaled, logger
    )

    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)

    # Calculate metrics
    if issubclass(type(model), ClassifierMixin):
        metrics = calculate_classification_metrics(
            model=model,
            x_train_scaled=x_train_scaled,
            x_test_scaled=x_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=y_train_pred,
            y_test_pred=y_test_pred,
            logger=logger,
        )
    elif issubclass(type(model), RegressorMixin):
        metrics = calculate_regression_metrics(
            y_train=y_train,
            y_test=y_test,
            y_train_pred=y_train_pred,
            y_test_pred=y_test_pred,
            logger=logger,
        )

    logger.info("Model successfully trained and metrics computed")
    logger.info("Logging the model on MLFlow")
    try:
        # Log the model on MLFlow
        log_on_mlflow(
            model_params,
            model_name,
            model,
            metrics,
            metadata,
            feature_importance_df,
            scaling_enable,
            scaler_file,
            scaler_bytes,
        )
        logger.info("Model %s successfully logged on MLflow", model_name)
    except Exception as e:
        logger.error("Error in logging the model in MLFlow server: %s", e)


def kfold_cross_validation(
    *,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    metadata: MetaData,
    models: dict[str, BaseEstimator],
    logger: Logger,
    n_splits: int = 5,
    scaling_enable: bool,
    scaler_file: str,
) -> None:
    """Function to perform K-Fold Cross Validation"""
    logger.info("Perform K-Fold Cross Validation")
    x = x_train
    y = y_train

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Dictionary to store scores
    mean_scores = {}
    # Get all estimators

    for model_name, model in models.items():
        # Scale x if scaling is enabled
        if scaling_enable:
            scaler = RobustScaler()
            x_scaled = scaler.fit_transform(x)
        else:
            x_scaled = x

        # Perform cross-validation
        scoring = "roc_auc" if issubclass(type(model), ClassifierMixin) else "r2"
        scores = cross_val_score(
            model, x_scaled, y.values.ravel(), cv=kf, scoring=scoring
        )

        # Store the scores in the dictionary
        mean_scores[model_name] = np.mean(scores)
        logger.debug(
            "Model: %s, Mean %s: %.4f, Std: %.4f",
            model_name,
            scoring,
            np.mean(scores),
            np.std(scores),
        )

    best_model_name = max(mean_scores, key=mean_scores.get)
    logger.info("K-Fold Cross Validation completed")
    model = models[best_model_name]
    train_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        metadata=metadata,
        model_name=best_model_name,
        model=model,
        scaling_enable=scaling_enable,
        scaler_file=scaler_file,
        logger=logger,
    )


def log_on_mlflow(
    model_params: dict,
    model_name: str,
    model: BaseEstimator,
    metrics: dict,
    metadata: MetaData,
    feature_importance_df: pd.DataFrame,
    scaling_enable,
    scaler_file: str,
    scaler_bytes: bytes,
):
    """Function to log the model on MLFlow"""
    # Logging on MLflow
    with mlflow.start_run():
        # Log the parameters
        mlflow.log_params(model_params)

        # Log the metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # Log the sklearn model and register
        mlflow.sklearn.log_model(
            signature=False,
            sk_model=model,
            artifact_path=model_name,
            registered_model_name=model_name,
            metadata=metadata.model_dump(),
        )

        # Add scaler file as model artifact if scaling is enabled
        if scaling_enable:
            scaler_b64 = base64.b64encode(scaler_bytes).decode("utf-8")
            mlflow.log_dict({"scaler": scaler_b64}, f"{model_name}/{scaler_file}")

        for key, value in metadata.model_dump().items():
            mlflow.set_tag(key, value)

        with NamedTemporaryFile(
            delete=True, prefix="feature_importance", suffix=".csv"
        ) as temp_file:
            feature_importance_df.to_csv(temp_file.name, index=False)
            mlflow.log_artifact(temp_file.name, artifact_path="feature_importance")


def split_and_clean_data(
    df: pd.DataFrame,
    *,
    start_col_x: int | None = None,
    end_col_x: int | None = None,
    start_col_y: int | None = None,
    end_col_y: int | None = None,
    settings: TrainingSettings,
):
    """Divide the dataset in training and test sets and remove outliers if enabled"""
    if start_col_y is None:
        start_col_y = end_col_x

    x_train, x_test, y_train, y_test = train_test_split(
        df.iloc[:, start_col_x:end_col_x],
        df.iloc[:, start_col_y:end_col_y],
        test_size=settings.TEST_SIZE,
        random_state=42,
    )

    if settings.REMOVE_OUTLIERS:
        x_train, y_train = remove_outliers(
            x_train,
            y_train,
            q1=settings.Q1_FACTOR,
            q3=settings.Q3_FACTOR,
            k=settings.THRESHOLD_FACTOR,
        )

    return x_train, x_test, y_train, y_test


def training_phase(
    *,
    models: list[BaseEstimator],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    metadata: MetaData,
    settings: TrainingSettings,
    logger: Logger,
):
    """Train the regression model chosen by the user or perform k-fold cross validation
    and then train the best model"""
    if len(models) == 1:
        model_name, model = next(iter(models.items()))
        train_model(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            metadata=metadata,
            model_name=model_name,
            model=model,
            scaling_enable=settings.SCALING_ENABLE,
            scaler_file=settings.SCALER_FILE,
            logger=logger,
        )
    else:
        kfold_cross_validation(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            metadata=metadata,
            models=models,
            n_splits=settings.KFOLDS,
            scaling_enable=settings.SCALING_ENABLE,
            scaler_file=settings.SCALER_FILE,
            logger=logger,
        )


def run(logger: Logger) -> None:
    """Function to load the dataset, do preprocessing, perform training and save the
    model on MLFlow"""

    # Load the settings and setup MLFlow
    try:
        settings = load_training_settings()
    except (ValueError, TypeError) as e:
        logger.error(e)
        exit(1)
    setup_mlflow(logger=logger)

    # Load the dataset and do preprocessing
    df = load_dataset(settings=settings, logger=logger)
    df = preprocessing(
        df=df,
        complex_templates=settings.TEMPLATE_COMPLEX_TYPES,
        logger=logger,
    )
    metadata = MetaData(
        start_time=df[DF_TIMESTAMP].max().strftime("%Y-%m-%d %H:%M:%S"),
        end_time=df[DF_TIMESTAMP].min().strftime("%Y-%m-%d %H:%M:%S"),
        features=settings.FINAL_FEATURES,
        features_number=len(settings.FINAL_FEATURES),
        remove_outliers=settings.REMOVE_OUTLIERS,
    )
    df = df[settings.FINAL_FEATURES]

    # Classification training phase
    logger.info("Classification phase started")
    x_train_cleaned, x_test, y_train_cleaned, y_test = split_and_clean_data(
        df, end_col_x=STATUS_COL, end_col_y=DEP_TIME_COL, settings=settings
    )

    training_phase(
        models=settings.CLASSIFICATION_MODELS,
        x_train=x_train_cleaned,
        x_test=x_test,
        y_train=y_train_cleaned,
        y_test=y_test,
        metadata=metadata,
        settings=settings,
        logger=logger,
    )
    logger.info("Classification phase ended")

    # Regression training phase
    logger.info("Regression phase started")
    x_train_cleaned, x_test, y_train_cleaned, y_test = split_and_clean_data(
        df, end_col_x=STATUS_COL, start_col_y=DEP_TIME_COL, settings=settings
    )

    training_phase(
        models=settings.REGRESSION_MODELS,
        x_train=x_train_cleaned,
        x_test=x_test,
        y_train=y_train_cleaned,
        y_test=y_test,
        metadata=metadata,
        settings=settings,
        logger=logger,
    )
    logger.info("Regression phase ended")
