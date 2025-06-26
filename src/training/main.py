import pickle
from logging import Logger

import numpy as np
import pandas as pd
import shap
from kafka.errors import NoBrokersAvailable
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier, is_regressor
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
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import RobustScaler

from src.processing import DF_TIMESTAMP, preprocessing
from src.settings import TrainingSettings, load_training_settings
from src.training.models import ClassificationMetrics, MetaData, RegressionMetrics
from src.utils import load_dataset_from_kafka_messages, load_local_dataset
from src.utils.kafka import create_kafka_consumer
from src.utils.mlflow import log_on_mlflow, setup_mlflow

SEED = 42


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
    return filtered[x.columns], filtered[y.columns]


def split_and_clean_data(
    *, x: pd.DataFrame, y: pd.DataFrame, settings: TrainingSettings
):
    """Divide the dataset in training and test sets and remove outliers if enabled.

    Remove outliers only from the train set.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=settings.TEST_SIZE, random_state=SEED
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


def get_feature_importance(
    model: BaseEstimator,
    columns: pd.Index,
    x_train_scaled: pd.DataFrame,
    logger: Logger,
) -> pd.DataFrame:
    """Calculate feature importance in different ways depending on the model"""

    if x_train_scaled.empty:
        logger.error("Input data 'x_train_scaled' is empty.")
        exit(1)

    # Case 1: Models with attribute `feature_importances_`
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        if len(feature_importances) != len(columns):
            logger.error(
                "Length mismatch: feature_importances_ has length %d, but columns has length %d",
                len(feature_importances),
                len(columns),
            )
            exit(1)
        feature_importance_df = pd.DataFrame(
            {"Feature": columns, "Importance": feature_importances}
        ).sort_values(by="Importance", ascending=False)
        logger.debug("Feature Importance:")
        logger.debug(feature_importance_df)
        return feature_importance_df

    # Case 2: Models like Lasso (coefficient)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        if len(coef) != len(columns):
            logger.error(
                "Length mismatch: coef_ has length %d, but columns has length %d",
                len(coef),
                len(columns),
            )
            exit(1)
        feature_importance_df = pd.DataFrame(
            {"Feature": columns, "Coefficient": coef}
        ).sort_values(by="Coefficient", ascending=False)
        logger.debug("Feature importance:")
        logger.debug(feature_importance_df)
        return feature_importance_df

    # Case 3: Models without `feature_importances_` or `coef_`, use SHAP
    else:
        try:
            background_data_summarized = shap.sample(x_train_scaled[:50])
            explainer = shap.KernelExplainer(
                model.predict_proba, background_data_summarized
            )
            shap_values = explainer.shap_values(x_train_scaled)
            # Compute feature importance
            shap_importance = np.mean(np.abs(shap_values), axis=0)[:, 0]
            if len(shap_importance) != len(columns):
                logger.error(
                    "Length mismatch: SHAP importance has length %d, but columns has length %d",
                    len(shap_importance),
                    len(columns),
                )
                exit(1)
            feature_importance_df = pd.DataFrame(
                {"Feature": columns, "Importance": shap_importance}
            ).sort_values(by="Importance", ascending=False)
            logger.debug("Feature importance:")
            logger.debug(feature_importance_df)
            return feature_importance_df
        except Exception as e:
            logger.error("Error in using SHAP: %s", e)
            exit(1)


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
    logger.info("Calculate classification metrics")
    if not hasattr(model, "predict_proba"):
        logger.error(
            "The model %s does not support predict_proba for AUC computation.",
            model.__class__.__name__,
        )
        exit(1)
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
    logger.info("Calculate regression metrics")
    train_metrics = RegressionMetrics(
        mse=float(mean_squared_error(y_train, y_train_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        mae=float(mean_absolute_error(y_train, y_train_pred)),
        r2=float(r2_score(y_train, y_train_pred)),
    )
    logger.debug("Train dataset metrics: %s", train_metrics)
    test_metrics = RegressionMetrics(
        mse=float(mean_squared_error(y_test, y_test_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        mae=float(mean_absolute_error(y_test, y_test_pred)),
        r2=float(r2_score(y_test, y_test_pred)),
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
    logger.info("Model successfully trained")

    logger.info("Computing metrics")
    # Get feature importance
    feature_importance_df = get_feature_importance(
        model, x_train.columns, x_train_scaled, logger
    )

    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)

    # Calculate metrics
    if is_classifier(model):
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
    elif is_regressor(model):
        metrics = calculate_regression_metrics(
            y_train=y_train,
            y_test=y_test,
            y_train_pred=y_train_pred,
            y_test_pred=y_test_pred,
            logger=logger,
        )
    logger.info("Metrics computed")

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
        logger=logger,
    )


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

    # Initialize StratifiedKFold/KFold
    first_model = next(iter(models.values()))
    if is_classifier(first_model):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    if is_regressor(first_model):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

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
        scoring = "roc_auc" if is_classifier(model) else "r2"
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
    logger.info("K-Fold Cross Validation completed. Best model is: %s", best_model_name)

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


def load_training_data(
    *,
    local_mode: bool = False,
    local_dataset: str | None = None,
    local_dataset_version: str = "1.1.0",
    kafka_server_url: str | None = None,
    kafka_topic: str | None = None,
    kafka_topic_partition: int = 0,
    kafka_topic_offset: int = 0,
    kafka_consumer_timeout_ms: int = 0,
    logger: Logger,
) -> pd.DataFrame:
    """Load the dataset from a local CSV file or from a kafka topic."""
    logger.info("Upload training data")
    if local_mode:
        if local_dataset:
            return load_local_dataset(
                filename=local_dataset,
                dataset_version=local_dataset_version,
                logger=logger,
            )
        raise ValueError("LOCAL_DATASET environment variable has not been set.")
    consumer = create_kafka_consumer(
        kafka_server_url=kafka_server_url,
        topic=kafka_topic,
        partition=kafka_topic_partition,
        offset=kafka_topic_offset,
        consumer_timeout_ms=kafka_consumer_timeout_ms,
        logger=logger,
    )
    df = load_dataset_from_kafka_messages(consumer=consumer, logger=logger)
    consumer.close()
    return df


def run(logger: Logger) -> None:
    """Function to load the dataset, do preprocessing, perform training and save the
    model on MLFlow"""

    # Load the settings and setup MLFlow
    settings = load_training_settings(logger=logger)
    setup_mlflow(logger=logger)

    # Load the training dataset
    try:
        df = load_training_data(
            local_mode=settings.LOCAL_MODE,
            local_dataset=settings.LOCAL_DATASET,
            local_dataset_version=settings.LOCAL_DATASET_VERSION,
            kafka_server_url=settings.KAFKA_HOSTNAME,
            kafka_topic=settings.KAFKA_TRAINING_TOPIC,
            kafka_topic_partition=settings.KAFKA_TRAINING_TOPIC_PARTITION,
            kafka_topic_offset=settings.KAFKA_TRAINING_TOPIC_OFFSET,
            kafka_consumer_timeout_ms=settings.KAFKA_TRAINING_TOPIC_TIMEOUT,
            logger=logger,
        )
    except FileNotFoundError:
        logger.error("File '%s' not found", settings.LOCAL_DATASET)
        exit(1)
    except NoBrokersAvailable:
        logger.error("Kakfa broker not found at given url: %s", settings.KAFKA_HOSTNAME)
        exit(1)
    except AssertionError as e:
        logger.error(e)
        exit(1)

    # Pre-process data
    df = preprocessing(
        df=df,
        complex_templates=settings.TEMPLATE_COMPLEX_TYPES,
        logger=logger,
    )
    if df.empty:
        logger.warning("No data to pre-process. No model generation.")
        return

    missing_features = set(
        settings.X_FEATURES
        + settings.Y_CLASSIFICATION_FEATURES
        + settings.Y_REGRESSION_FEATURES
    ).difference(set(df.columns))
    if len(missing_features) > 0:
        logger.error(
            "Given final features are not present in the training data: %s",
            missing_features,
        )
        exit(1)
    start_timestamp: pd.Timestamp = df[DF_TIMESTAMP].min()
    end_timestamp: pd.Timestamp = df[DF_TIMESTAMP].max()

    # Classification training phase
    logger.info("Classification phase started")
    metadata = MetaData(
        start_time=start_timestamp.isoformat(),
        end_time=end_timestamp.isoformat(),
        features=settings.X_FEATURES + settings.Y_CLASSIFICATION_FEATURES,
        remove_outliers=settings.REMOVE_OUTLIERS,
        scaling=settings.SCALING_ENABLE,
        scaler_file=settings.SCALER_FILE,
    )
    x_train_cleaned, x_test, y_train_cleaned, y_test = split_and_clean_data(
        x=df[settings.X_FEATURES],
        y=df[settings.Y_CLASSIFICATION_FEATURES],
        settings=settings,
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
    metadata = MetaData(
        start_time=start_timestamp.isoformat(),
        end_time=end_timestamp.isoformat(),
        features=settings.X_FEATURES + settings.Y_REGRESSION_FEATURES,
        remove_outliers=settings.REMOVE_OUTLIERS,
        scaling=settings.SCALING_ENABLE,
        scaler_file=settings.SCALER_FILE,
    )
    x_train_cleaned, x_test, y_train_cleaned, y_test = split_and_clean_data(
        x=df[settings.X_FEATURES],
        y=df[settings.Y_REGRESSION_FEATURES],
        settings=settings,
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
