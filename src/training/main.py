from logging import Logger
from tempfile import NamedTemporaryFile

import mlflow
import mlflow.environment_variables
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import pickle
from sklearn.base import ClassifierMixin, RegressorMixin
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
from sklearn.utils import all_estimators

from processing import DF_TIMESTAMP, load_dataset, preprocessing
from settings import load_training_settings, setup_mlflow
from training.models import ClassificationMetrics, MetaData, RegressionMetrics

STATUS_COL = -2
DEP_TIME_COL = -1

single_vm = [
    "single-vm/single_vm.yaml",
    "single-vm/single_vm_with_volume.yaml",
    "single-vm/private-net/single_vm.yaml",
    "single-vm/private-net/single_vm_with_volume.yaml",
]
single_vm_complex = [
    "single-vm/cloud_storage_service.yaml",
    "single-vm/elasticsearch_kibana.yaml",
    "single-vm/iam_voms-aa.yaml",
]
k8s = [
    "kubernetes/k8s_cluster.yaml",
    "kubernetes/k8s_cluster_with_addons.yaml",
    "kubernetes/htcondor_k8s.yaml",
    "kubernetes/private-net/k8s_cluster.yaml",
    "kubernetes/spark_cluster.yaml",
]
docker = [
    "docker/run_docker.yaml",
    "docker/docker_compose.yaml",
    "docker/docker_compose_with_volume.yaml",
    "docker/run_docker_with_volume.yaml",
]
jupyter = [
    "jupyter/jupyter_vm.yaml",
    "jupyter/jupyter_matlab.yaml",
    "jupyter/ml_infn.yaml",
    "jupyter/cygno_experiment.yaml",
    "jupyter/private-net/jupyter_vm.yaml ",
]
all = single_vm + single_vm_complex + k8s + docker + jupyter
simple = single_vm + docker
complex = single_vm_complex + k8s + jupyter


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


def get_feature_importance(model, columns, x_train_scaled, logger):
    # Case 1: Models with attribute `feature_importances_`
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame(
            {"Feature": columns, "Importance": feature_importances}
        ).sort_values(by="Importance", ascending=False)
        logger.debug("Important features:")
        logger.debug(feature_importance_df)
        return feature_importance_df

    # Case 2: Models like Lasso (coefficient)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        feature_importance_df = pd.DataFrame(
            {"Feature": columns, "Coefficient": coef}
        ).sort_values(by="Coefficient", ascending=False)
        logger.debug("Important features:")
        logger.debug(feature_importance_df)
        return feature_importance_df

    # Case 3: Models without `feature_importances_` or `coef_`, use SHAP
    else:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_train_scaled)
            shap.summary_plot(shap_values, x_train_scaled)
            # Gives mean values of feayure importance
            shap_importance = np.mean(np.abs(shap_values), axis=0)
            feature_importance_df = pd.DataFrame(
                {"Feature": columns, "Importance": shap_importance}
            ).sort_values(by="Importance", ascending=False)
            logger.debug("Important features:")
            logger.debug(feature_importance_df)
            return feature_importance_df
        except Exception as e:
            print(f"Errore nell'uso di SHAP: {e}")
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
    logger.debug("Train dataset metrics: %s", train_metrics)
    test_metrics = ClassificationMetrics(
        accuracy=accuracy_score(y_test, y_test_pred),
        auc=roc_auc_score(y_test, model.predict_proba(x_test_scaled)[:, 1]),
        f1=f1_score(y_test, y_test_pred, average="binary"),
        precision=precision_score(y_test, y_test_pred, average="binary"),
        recall=recall_score(y_test, y_test_pred, average="binary"),
    )
    logger.debug("Test dataset metrics: %s", test_metrics)

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
    model_params: dict,
    scaler_file: str,
    logger: Logger,
) -> None:
    """
    Function to train a generic sklearn ML model

    :param model_name: Name of the model to train (e.g. 'RandomForestClassifier').
    :param model_params: Parameters to pass to the model as a dictionary.
    :param experiment_name: Name of the MLFlow experiment
    """

    # Dinamically load the model with the requested parameters
    try:
        cls = dict(all_estimators()).get(model_name, None)
        model = cls(**model_params)
    except TypeError as e:
        logger.error("Error in '%s' model creation: %s", model_name, e)
        exit(1)

    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    scaler_bytes = pickle.dumps(scaler)
    x_test_scaled = scaler.transform(x_test)

    # Train the model
    logger.info("Training model '%s' with params: %s", model_name, model_params)
    model.fit(x_train_scaled, y_train.values.ravel())

    # Get feature importance
    feature_importance_df = get_feature_importance(
        model, x_train.columns, x_train_scaled, logger
    )

    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)

    if issubclass(cls, ClassifierMixin):
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
    elif issubclass(cls, RegressorMixin):
        metrics = calculate_regression_metrics(
            y_train=y_train,
            y_test=y_test,
            y_train_pred=y_train_pred,
            y_test_pred=y_test_pred,
            logger=logger,
        )

    log_on_mlflow(
        model_params, model_name, model, metrics, metadata, feature_importance_df, scaler_file, scaler_bytes
    )


def kfold_cross_validation(
    *,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    metadata: MetaData,
    models_params: dict[str, dict],
    logger: Logger,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    scaler_file: str,
) -> None:
    x = x_train
    y = y_train

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Dictionary to store scores
    model_scores = {}
    mean_scores = {}
    # Get all estimators
    all_models = dict(all_estimators())

    # Initialize RobustScaler
    scaler = RobustScaler()

    for model_name, params in models_params.items():
        # Fetch the model class
        ModelClass = all_models.get(model_name)
        if ModelClass is None:
            raise ValueError(f"Model {model_name} not found in sklearn estimators.")

        # Instantiate the model with parameters
        model = ModelClass(**params)

        # Scale the features
        x_scaled = scaler.fit_transform(x)

        # Perform cross-validation
        scores = cross_val_score(
            model, x_scaled, y.values.ravel(), cv=kf, scoring=scoring
        )

        # Store the scores in the dictionary
        model_scores[model_name] = scores
        mean_scores[model_name] = np.mean(scores)
        print(
            f"Model: {model_name}, Mean {scoring}: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}"
        )
    best_model_name = max(mean_scores, key=mean_scores.get)
    print(f"Model selected for training: {best_model_name}")
    train_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        metadata=metadata,
        model_name=best_model_name,
        model_params=models_params[best_model_name],
        scaler_file=scaler_file,
        logger=logger,
    )

    return model_scores


def log_on_mlflow(
    model_params: dict,
    model_name: str,
    model: any,
    metrics: dict,
    metadata: MetaData,
    feature_importance_df: pd.DataFrame,
    scaler_file: str,
    scaler_bytes: bytes,
):
    # Logging on MLflow
    with mlflow.start_run():
        # Log the parameters
        mlflow.log_params(model_params)

        # Print the metrics
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        # Log the metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        #Log the sklearn model and register
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            registered_model_name=model_name,
            metadata=metadata.model_dump(),
        )

        with open("/tmp/scaler.pkl", "wb") as f:  # Salva lo scaler temporaneamente
            f.write(scaler_bytes)
        mlflow.log_artifact(local_path="/tmp/scaler.pkl", artifact_path=model_name)
        for key, value in metadata.model_dump().items():
            mlflow.set_tag(key, value)

        import os
        os.remove("/tmp/scaler.pkl")

        with NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            feature_importance_df.to_csv(temp_file.name, index=False)
            mlflow.log_artifact(temp_file.name, artifact_path="feature_importances")
    print(f"Model {model_name} successfully logged on MLflow.")


def run(logger: Logger) -> None:
    settings = load_training_settings()
    setup_mlflow(logger=logger)

    df = load_dataset(settings=settings, logger=logger)
    metadata = MetaData(
        start_time=df[DF_TIMESTAMP].max(),
        end_time=df[DF_TIMESTAMP].min(),
        features=settings.FINAL_FEATURES,
        features_number=len(settings.FINAL_FEATURES),
        remove_outliers=settings.REMOVE_OUTLIERS,
    )
    df = preprocessing(
        df=df,
        complex_templates=settings.TEMPLATE_COMPLEX_TYPES,
        logger=logger,
    )
    df = df[settings.FINAL_FEATURES]

    x_train, x_test, y_train, y_test = train_test_split(
        df.iloc[:, :STATUS_COL],
        df.iloc[:, STATUS_COL:DEP_TIME_COL],
        test_size=settings.TEST_SIZE,
        random_state=42,
    )
    if settings.REMOVE_OUTLIERS:
        x_train_cleaned, y_train_cleaned = remove_outliers(
            x_train,
            y_train,
            q1=settings.Q1_FACTOR,
            q3=settings.Q3_FACTOR,
            k=settings.THRESHOLD_FACTOR,
        )
    else:
        x_train_cleaned, y_train_cleaned = x_train, y_train

    # Train the classification model chosen by the user
    # or perform k-fold cross validation
    if len(settings.CLASSIFICATION_MODELS.keys()) == 1:
        model = next(iter(settings.CLASSIFICATION_MODELS))
        train_model(
            x_train=x_train_cleaned,
            x_test=x_test,
            y_train=y_train_cleaned,
            y_test=y_test,
            metadata=metadata,
            model_name=model,
            model_params=settings.CLASSIFICATION_MODELS.get(model),
            scaler_file=settings.SCALER_FILE,
            logger=logger,
        )
    else:
        # Perform KFold cross validation
        kfold_cross_validation(
            x_train=x_train_cleaned,
            x_test=x_test,
            y_train=y_train_cleaned,
            y_test=y_test,
            metadata=metadata,
            models_params=settings.CLASSIFICATION_MODELS,
            n_splits=settings.KFOLDS,
            scoring="roc_auc",
            scaler_file=settings.SCALER_FILE,
            logger=logger,
        )

    x_train, x_test, y_train, y_test = train_test_split(
        df.iloc[:, :STATUS_COL],
        df.iloc[:, DEP_TIME_COL:],
        test_size=settings.TEST_SIZE,
        random_state=42,
    )
    if settings.REMOVE_OUTLIERS:
        x_train_cleaned, y_train_cleaned = remove_outliers(
            x_train,
            y_train,
            q1=settings.Q1_FACTOR,
            q3=settings.Q3_FACTOR,
            k=settings.THRESHOLD_FACTOR,
        )
    else:
        x_train_cleaned, y_train_cleaned = x_train, y_train

    # Train the regression model chosen by the user
    # or perform k-fold cross validation
    if len(settings.REGRESSION_MODELS.keys()) == 1:
        model = next(iter(settings.REGRESSION_MODELS))
        train_model(
            x_train=x_train_cleaned,
            x_test=x_test,
            y_train=y_train_cleaned,
            y_test=y_test,
            metadata=metadata,
            model_name=model,
            model_params=settings.REGRESSION_MODELS.get(model),
            scaler_file=settings.SCALER_FILE,
            logger=logger,
        )
    else:
        kfold_cross_validation(
            x_train=x_train_cleaned,
            x_test=x_test,
            y_train=y_train_cleaned,
            y_test=y_test,
            metadata=metadata,
            models_params=settings.REGRESSION_MODELS,
            n_splits=settings.KFOLDS,
            scoring="r2",
            scaler_file=settings.SCALER_FILE,
            logger=logger,
        )