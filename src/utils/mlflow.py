import base64
import pickle
import time
from logging import Logger
from tempfile import NamedTemporaryFile

import mlflow
import mlflow.environment_variables
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel, load_model
from sklearn.base import BaseEstimator
from sklearn.preprocessing import RobustScaler

from settings import load_mlflow_settings
from training.models import MetaData


def setup_mlflow(*, logger: Logger) -> None:
    """Function to set up the mlflow settings"""
    logger.info("Setting up MLFlow service communication")
    settings = load_mlflow_settings(logger=logger)
    try:
        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_TIMEOUT.set(
            settings.MLFLOW_HTTP_REQUEST_TIMEOUT
        )
        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_MAX_RETRIES.set(
            settings.MLFLOW_HTTP_REQUEST_MAX_RETRIES
        )
        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR.set(
            settings.MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR
        )
        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_BACKOFF_JITTER.set(
            settings.MLFLOW_HTTP_REQUEST_BACKOFF_JITTER
        )

        mlflow.set_tracking_uri(str(settings.MLFLOW_TRACKING_URI))
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    except mlflow.exceptions.MlflowException as e:
        logger.error(e.message)
        exit(1)


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
    max_retries: int,
    attempt_interval: float,
    logger: Logger,
) -> bool:
    """Function to log the model on MLFlow"""
    attempt = 0
    success = False
    logger.info("Logging the model on MLFlow")
    while not success:
        try:
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
                    mlflow.log_dict(
                        {"scaler": scaler_b64}, f"{model_name}/{scaler_file}"
                    )

                for key, value in metadata.model_dump().items():
                    mlflow.set_tag(key, value)

                with NamedTemporaryFile(
                    delete=True, prefix="feature_importance", suffix=".csv"
                ) as temp_file:
                    feature_importance_df.to_csv(temp_file.name, index=False)
                    mlflow.log_artifact(
                        temp_file.name, artifact_path="feature_importance"
                    )
            logger.info("Model %s successfully logged on MLflow", model_name)
            success = True
        except MlflowException:
            if attempt >= max_retries:
                logger.error("Error in logging the model in MLFlow server")
                exit(1)
            attempt += 1
            logger.warning(
                "Can't contact MLFlow server (%d/%d). Retrying in %s seconds",
                attempt,
                max_retries,
                attempt_interval,
            )
            time.sleep(attempt_interval)


def get_model_uri(
    client: mlflow.MlflowClient, *, model_name: str, model_version: str | int
) -> str:
    """Get target or latest model version from MLFlow."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if len(versions) == 0:
        raise ValueError(f"Model '{model_name}' not found")
    if len(versions) == 1:
        version = versions[0]
    elif model_version == "latest":
        version = max(versions, key=lambda x: int(x.version))
    else:
        version = next(
            filter(lambda x: int(x.version) == model_version, versions), None
        )
    if version is None:
        raise ValueError(f"Version {model_version} for model '{model_name}' not found")
    return f"models:/{version.name}/{version.version}"


def get_model(*, model_uri: str) -> tuple[PyFuncModel, BaseEstimator]:
    """Get the model type and load it with the proper function"""
    model = load_model(model_uri)
    model_type = model.metadata.flavors.keys()
    if model.loader_module == "mlflow.sklearn":
        try:
            return model, mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            raise ValueError(f"Model not found at given uri '{model_uri}'") from e
    raise ValueError(f"Model {model_type} not in the mlflow.sklearn library")


def get_scaler(*, model_uri: str, scaler_file: str) -> RobustScaler:
    """Return model's scaler"""
    scaler_uri = f"{model_uri}/{scaler_file}"
    scaler_dict = mlflow.artifacts.load_dict(scaler_uri)
    scaler_bytes = base64.b64decode(scaler_dict["scaler"])
    return pickle.loads(scaler_bytes)
