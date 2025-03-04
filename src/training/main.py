from logging import Logger
from tempfile import NamedTemporaryFile

import mlflow
import mlflow.environment_variables
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
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
from training.models import MetaData

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


def train_model_classification(
    *,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    metadata: MetaData,
    model_name: str,
    model_params: dict,
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
        logger.error("Error in the creation of the model: %s", e)
        exit(1)

    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train the model
    logger.info("Training model '%s' with params: %s", model_name, model_params)
    model.fit(x_train_scaled, y_train.values.ravel())

    # Get feature importance
    feature_importance_df = get_feature_importance(
        model, x_train.columns, x_train_scaled
    )

    if feature_importance_df is not None:
        print(feature_importance_df)

    # Do predictions
    y_pred_train = model.predict(x_train_scaled)
    y_pred_test = model.predict(x_test_scaled)

    # Compute the accuracy if the model is a classifier
    if isinstance(model, ClassifierMixin):
        metrics = {
            "Accuracy train": accuracy_score(y_train, y_pred_train),
            "auc train": roc_auc_score(
                y_train, model.predict_proba(x_train_scaled)[:, 1]
            ),
            "F1 train": f1_score(y_train, y_pred_train, average="binary"),
            "Precision train": precision_score(y_train, y_pred_train, average="binary"),
            "Recall train": recall_score(y_train, y_pred_train, average="binary"),
            "Accuracy test": accuracy_score(y_test, y_pred_test),
            "auc test": roc_auc_score(y_test, model.predict_proba(x_test_scaled)[:, 1]),
            "F1 test": f1_score(y_test, y_pred_test, average="binary"),
            "Precision test": precision_score(y_test, y_pred_test, average="binary"),
            "Recall test": recall_score(y_test, y_pred_test, average="binary"),
        }
        print(confusion_matrix(y_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))
        log_on_mlflow(
            model_params, model_name, model, metrics, metadata, feature_importance_df
        )


def train_model_regression(
    *,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    metadata: MetaData,
    model_name: str,
    model_params: dict,
    logger: Logger,
) -> None:
    # Load the model dinamically
    ModelClass = dict(all_estimators())[model_name]

    # Create the model with the requested parameters
    try:
        model = ModelClass(**model_params)
    except TypeError as e:
        print(f"Error in the creation of the model: {e}")
        return

    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train the model
    model.fit(x_train_scaled, y_train.values.ravel())

    # Get feature importance
    feature_importance_df = get_feature_importance(
        model, x_train.columns, x_train_scaled, logger
    )

    if feature_importance_df is not None:
        print(feature_importance_df)

    # Do predictions
    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)

    if isinstance(model, RegressorMixin):
        # Compute metrics
        metrics = {
            "MSE train": mean_squared_error(y_train, y_train_pred),
            "MSE test": mean_squared_error(y_test, y_test_pred),
            "RMSE train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "RMSE test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "MAE train": mean_absolute_error(y_train, y_train_pred),
            "MAE test": mean_absolute_error(y_test, y_test_pred),
            "R2 train": r2_score(y_train, y_train_pred),
            "R2 test": r2_score(y_test, y_test_pred),
        }

        log_on_mlflow(
            model_params, model_name, model, metrics, metadata, feature_importance_df
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
    if issubclass(all_models.get(best_model_name), ClassifierMixin):
        train_model_classification(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            metadata=metadata,
            model_name=best_model_name,
            model_params=models_params[best_model_name],
            logger=logger
        )
    elif issubclass(all_models.get(best_model_name), RegressorMixin):
        train_model_regression(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            metadata=metadata,
            model_name=best_model_name,
            model_params=models_params[best_model_name],
            logger=logger
        )

    return model_scores


def log_on_mlflow(
    model_params: dict,
    model_name: str,
    model: any,
    metrics: dict,
    metadata: MetaData,
    feature_importance_df: pd.DataFrame,
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

        # Log the sklearn model and register
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            registered_model_name=model_name,
            metadata=metadata.model_dump(),
        )
        for key, value in metadata.model_dump().items():
            mlflow.set_tag(key, value)

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
        final_features=settings.FINAL_FEATURES,
        logger=logger,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        df.iloc[:, :-2],
        df.iloc[:, -2:-1],
        test_size=settings.TEST_SIZE,
        random_state=42,
    )
    if settings.REMOVE_OUTLIERS:
        x_train_cleaned, y_train_cleaned = remove_outliers(x_train, y_train)
    else:
        x_train_cleaned, y_train_cleaned = x_train, y_train

    # Train the classification model chosen by the user
    # or perform k-fold cross validation
    if len(settings.CLASSIFICATION_MODELS.keys()) == 1:
        model = settings.CLASSIFICATION_MODELS.keys()[0]
        train_model_classification(
            x_train=x_train_cleaned,
            x_test=x_test,
            y_train=y_train_cleaned,
            y_test=y_test,
            metadata=metadata,
            model_name=model,
            model_params=settings.CLASSIFICATION_MODELS.get(model),
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
            logger=logger,
        )

    # Train the regression model chosen by the user
    # or perform k-fold cross validation
    if len(settings.REGRESSION_MODELS.keys()) == 1:
        model = settings.REGRESSION_MODELS.keys()[0]
        train_model_regression(
            x_train=x_train_cleaned,
            x_test=x_test,
            y_train=y_train_cleaned,
            y_test=y_test,
            metadata=metadata,
            model_name=model,
            model_params=settings.REGRESSION_MODELS.get(model),
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
            logger=logger,
        )
