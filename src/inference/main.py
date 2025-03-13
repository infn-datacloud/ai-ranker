import base64
import json
import pickle
import time
from logging import Logger
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.pyfunc import PyFuncModel, load_model
from sklearn.base import BaseEstimator

from processing import (
    DF_AVG_FAIL_TIME,
    DF_AVG_SUCCESS_TIME,
    DF_COMPLEX,
    DF_CPU_DIFF,
    DF_DISK_DIFF,
    DF_FAIL_PERC,
    DF_GPU,
    DF_INSTANCE_DIFF,
    DF_PROVIDER,
    DF_PUB_IPS_DIFF,
    DF_RAM_DIFF,
    DF_TIMESTAMP,
    MSG_CPU_QUOTA,
    MSG_CPU_REQ,
    MSG_CPU_USAGE,
    MSG_DISK_QUOTA,
    MSG_DISK_REQ,
    MSG_DISK_USAGE,
    MSG_GPU_REQ,
    MSG_INSTANCE_QUOTA,
    MSG_INSTANCE_REQ,
    MSG_INSTANCE_USAGE,
    MSG_INSTANCES_WITH_EXACT_FLAVORS,
    MSG_PROVIDER_NAME,
    MSG_PUB_IPS_QUOTA,
    MSG_PUB_IPS_REQ,
    MSG_PUB_IPS_USAGE,
    MSG_RAM_QUOTA,
    MSG_RAM_REQ,
    MSG_RAM_USAGE,
    MSG_REGION_NAME,
    MSG_TEMPLATE_NAME,
    load_dataset,
    preprocessing,
)
from settings import (
    InferenceSettings,
    create_kafka_consumer,
    create_kafka_producer,
    load_inference_settings,
    setup_mlflow,
)

CLASSIFICATION_SUCCESS_IDX = 0
# Default value for unexpected responses
DEFAULT_PROBABILITY = 0.1
K_RES_EXACT = "resource_exactness"
K_CLASS = "classification"
K_REGR = "regression"
EXAMPLE = [
    json.dumps(
        {
            "msg_version": "1.0.0",
            "providers": [
                {
                    "exact_flavors": 1.0,
                    "floating_ips_quota": 100.0,
                    "floating_ips_requ": 1.0,
                    "floating_ips_usage": 72.0,
                    "gpus_requ": 0.0,
                    "overbooking_ram": 1.2,
                    "overbooking_cpu": 4.0,
                    "n_instances_quota": 500.0,
                    "n_instances_requ": 2.0,
                    "n_instances_usage": 102.0,
                    "n_volumes_quota": 500.0,
                    "n_volumes_requ": 0.0,
                    "n_volumes_usage": 42.0,
                    "provider_name": "CLOUD-INFN-CATANIA",
                    "ram_gb_quota": 8000.0,
                    "ram_gb_requ": 8.0,
                    "ram_gb_usage": 590.0,
                    "region_name": "INFN-CT",
                    "storage_gb_quota": 10000.0,
                    "storage_gb_requ": 0.0,
                    "storage_gb_usage": 2375.0,
                    "test_failure_perc_1d": 0.0,
                    "test_failure_perc_30d": 0.009,
                    "test_failure_perc_7d": 0.012,
                    "vcpus_quota": 1000.0,
                    "vcpus_requ": 4.0,
                    "vcpus_usage": 295.0,
                },
                {
                    "exact_flavors": 0.0,
                    "floating_ips_quota": 100.0,
                    "floating_ips_requ": 1.0,
                    "floating_ips_usage": 72.0,
                    "gpus_requ": 0.0,
                    "overbooking_ram": 1.2,
                    "overbooking_cpu": 4.0,
                    "n_instances_quota": 500.0,
                    "n_instances_requ": 2.0,
                    "n_instances_usage": 102.0,
                    "n_volumes_quota": 500.0,
                    "n_volumes_requ": 0.0,
                    "n_volumes_usage": 42.0,
                    "provider_name": "CLOUD-VENETO",
                    "ram_gb_quota": 8000.0,
                    "ram_gb_requ": 8.0,
                    "ram_gb_usage": 590.0,
                    "region_name": "regionOne",
                    "storage_gb_quota": 10000.0,
                    "storage_gb_requ": 0.0,
                    "storage_gb_usage": 2375.0,
                    "test_failure_perc_1d": 0.0,
                    "test_failure_perc_30d": 0.009,
                    "test_failure_perc_7d": 0.012,
                    "vcpus_quota": 1000.0,
                    "vcpus_requ": 4.0,
                    "vcpus_usage": 295.0,
                },
            ],
            "template_name": "Cluster Kubernetes",
            "timestamp": "2025-02-03 15:58:44.068000",
            "user_group": "admins/catchall",
            "uuid": "11efe247-bea1-933a-8c8c-0242e34b7d6d",
        }
    )
]


def get_model_uri(
    client: MlflowClient, *, model_name: str, model_version: str | int
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


def get_scaler(*, model_uri: str, scaler_file: str) -> Any:
    """Return model's scaler"""
    scaler_uri = f"{model_uri}/{scaler_file}"
    scaler_dict = mlflow.artifacts.load_dict(scaler_uri)
    scaler_bytes = base64.b64decode(scaler_dict["scaler"])
    return pickle.loads(scaler_bytes)


def create_inference_df(*, data: dict, model: Any, scaler: Any | None) -> pd.DataFrame:
    """From received dict create the dataframe to send to inference.

    Filter features.
    """
    input_features = model.feature_names_in_
    values = [[data[k] for k in input_features if k in data]]
    x_new = pd.DataFrame(values, columns=input_features)
    if scaler is not None:
        return pd.DataFrame(
            scaler.transform(x_new), columns=x_new.columns, index=x_new.index
        )
    return x_new


def classification_predict(
    *,
    input_data: dict,
    model_name: str,
    model_version: str,
    scaler_file: str | None,
    mlflow_client: MlflowClient,
    logger: Logger,
):
    """Load model and predict result.

    TODO: ...
    """
    # Load model and scaler
    logger.info("Load model from MLFlow.")
    model_uri = get_model_uri(
        mlflow_client, model_name=model_name, model_version=model_version
    )
    mlflow_model, sklearn_model = get_model(model_uri=model_uri)
    scaler = None
    if scaler_file:
        logger.info("Load scaler from MLFlow.")
        scaler = get_scaler(model_uri=model_uri, scaler_file=scaler_file)

    # Calculate success probability for each provider
    classification_values = {}
    for provider, data in input_data.items():
        x_new = create_inference_df(data=data, model=sklearn_model, scaler=scaler)
        logger.info("Predict success probability for '%s'", provider)
        y_pred_new = sklearn_model.predict_proba(x_new)
        success_prob = float(y_pred_new[0][CLASSIFICATION_SUCCESS_IDX])
        logger.debug(
            "Predicted success probability for '%s': %.2f", provider, success_prob
        )
        # Keep only success probability
        classification_values[provider] = success_prob
    return classification_values


def regression_predict(
    *,
    input_data: dict,
    model_name: str,
    model_version: str,
    scaler_file: str | None,
    min_regression_time: float,
    max_regression_time: float,
    mlflow_client: MlflowClient,
    logger: Logger,
):
    """Load model and predict result.

    TODO: ...
    """
    # Load model and scaler
    logger.info("Load model from MLFlow.")
    model_uri = get_model_uri(
        mlflow_client, model_name=model_name, model_version=model_version
    )
    mlflow_model, sklearn_model = get_model(model_uri=model_uri)
    scaler = None
    if scaler_file:
        logger.info("Load scaler from MLFlow.")
        scaler = get_scaler(model_uri=model_uri, scaler_file=scaler_file)

    # Calculate expected time for each provider
    regression_values = {}
    for provider, data in input_data.items():
        x_new = create_inference_df(data=data, model=sklearn_model, scaler=scaler)
        logger.info("Predict time for '%s'", provider)
        y_pred_new = sklearn_model.predict(x_new)
        expected_time = float(y_pred_new[0])
        # Convert this value into a value between (0,1) range.
        regression_value = 1 - (expected_time - min_regression_time) / (
            max_regression_time - min_regression_time
        )
        logger.debug("Predicted time for '%s': %.2f", provider, expected_time)
        logger.debug("Predicted goodness for '%s': %.2f", provider, expected_time)
        regression_values[provider] = regression_value
    return regression_values


def process_inference(
    *,
    input_inference: dict,
    mlflow_client: MlflowClient,
    settings: InferenceSettings,
    logger: Logger,
) -> dict:
    start_time = time.time()

    # Send requests to classification and regression endpoints
    classification_response = classification_predict(
        input_data=input_inference,
        model_name=settings.CLASSIFICATION_MODEL_NAME,
        model_version=settings.CLASSIFICATION_MODEL_VERSION,
        scaler_file=settings.SCALER_FILE if settings.SCALING_ENABLE else None,
        mlflow_client=mlflow_client,
        logger=logger,
    )
    regression_response = regression_predict(
        input_data=input_inference,
        model_name=settings.REGRESSION_MODEL_NAME,
        model_version=settings.REGRESSION_MODEL_VERSION,
        scaler_file=settings.SCALER_FILE if settings.SCALING_ENABLE else None,
        min_regression_time=settings.REGRESSION_MIN_TIME,
        max_regression_time=settings.REGRESSION_MAX_TIME,
        mlflow_client=mlflow_client,
        logger=logger,
    )
    assert len(classification_response) == len(regression_response), (
        "Responses length mismatch."
    )

    results = {}
    for provider in input_inference.keys():
        results[provider] = {
            K_CLASS: classification_response[provider] * settings.CLASSIFICATION_WEIGHT,
            K_REGR: regression_response[provider]
            * (1 - settings.CLASSIFICATION_WEIGHT),
            K_RES_EXACT: input_inference[provider][K_RES_EXACT],
        }
    logger.debug("Results: %s", results)

    if settings.FILTER:
        results = {
            key: value
            for key, value in results.items()
            if value[K_CLASS] + value[K_REGR] >= settings.THRESHOLD
        }
        logger.debug(
            "Results with total succes greater then %.2f: %s",
            settings.THRESHOLD,
            results,
        )

    if settings.EXACT_RESOURCES_PRECEDENCE:
        # Sort results by resource exactness and then success probability

        def sort_key(item):
            return item[1][K_RES_EXACT], item[1][K_CLASS] + item[1][K_REGR]
    else:

        def sort_key(item):
            return item[1][K_CLASS] + item[1][K_REGR]

    sorted_results = dict(sorted(results.items(), key=sort_key, reverse=True))
    logger.debug("Sorted results: %s", results)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug("Inference Time: %.2f seconds", elapsed_time)
    return sorted_results


def pre_process_message(
    *,
    message: dict[str, Any],
    df: pd.DataFrame,
    complex_templates: list[str],
    logger: Logger,
) -> dict[str, Any]:
    """Create a dict with the relevant values calculated from the received message.

    For each provider, create a dict with the user requested values and the current
    provider situation inferred from the stored data.
    """
    data = {}
    template_name = message[MSG_TEMPLATE_NAME]
    complexity = 1 if template_name in complex_templates else 0
    for el in message["providers"]:
        # Filter historical values related to the target provider and region
        provider_region = f"{el[MSG_PROVIDER_NAME]}-{el[MSG_REGION_NAME]}"
        df_filtered = df[
            (df[MSG_TEMPLATE_NAME] == template_name)
            & (df[DF_PROVIDER] == provider_region)
        ]
        avg_success_time = 0.0
        avg_failure_time = 0.0
        failure_percentage = 0.0
        # Retrieve latest details
        if not df_filtered.empty:
            idx_latest = df_filtered[DF_TIMESTAMP].idxmax()
            avg_success_time = float(df_filtered.loc[idx_latest, DF_AVG_SUCCESS_TIME])
            avg_failure_time = float(df_filtered.loc[idx_latest, DF_AVG_FAIL_TIME])
            failure_percentage = float(df_filtered.loc[idx_latest, DF_FAIL_PERC])

        calculated_values = {
            DF_CPU_DIFF: (el[MSG_CPU_QUOTA] - el[MSG_CPU_REQ]) - el[MSG_CPU_USAGE],
            DF_RAM_DIFF: (el[MSG_RAM_QUOTA] - el[MSG_RAM_REQ]) - el[MSG_RAM_USAGE],
            DF_DISK_DIFF: (el[MSG_DISK_QUOTA] - el[MSG_DISK_REQ]) - el[MSG_DISK_USAGE],
            DF_INSTANCE_DIFF: (el[MSG_INSTANCE_QUOTA] - el[MSG_INSTANCE_REQ])
            - el[MSG_INSTANCE_USAGE],
            DF_PUB_IPS_DIFF: (el[MSG_PUB_IPS_QUOTA] - el[MSG_PUB_IPS_REQ])
            - el[MSG_PUB_IPS_USAGE],
            DF_GPU: float(bool(el[MSG_GPU_REQ])),
            DF_COMPLEX: complexity,
            "test_failure_perc_30d": el["test_failure_perc_30d"],
            "test_failure_perc_7d": el["test_failure_perc_7d"],
            "test_failure_perc_1d": el["test_failure_perc_1d"],
            "overbooking_ram": el["overbooking_ram"],
            "overbooking_cpu": el["overbooking_cpu"],
            DF_AVG_SUCCESS_TIME: avg_success_time,
            DF_AVG_FAIL_TIME: avg_failure_time,
            DF_FAIL_PERC: failure_percentage,
            K_RES_EXACT: el[MSG_INSTANCES_WITH_EXACT_FLAVORS] / el[MSG_INSTANCE_REQ],
        }
        data[provider_region] = calculated_values
        logger.debug(
            "The following data has been requested on provider '%s' on region '%s': %s",
            el[MSG_PROVIDER_NAME],
            el[MSG_REGION_NAME],
            calculated_values,
        )

    return data


def create_message(
    *, sorted_results: dict, input_data: dict, deployment_uuid: str, logger: Logger
) -> dict[str, Any]:
    """Create a dict with the deployment uuid and the list of ranked providers."""
    ranked_providers = []
    for provider, values in sorted_results.items():
        for i in input_data:
            if f"{i[MSG_PROVIDER_NAME]}-{i[MSG_REGION_NAME]}" == provider:
                data = {**values, **i}
                break
        ranked_providers.append(data)
    message = {"uuid": deployment_uuid, "ranked_providers": ranked_providers}
    logger.debug("Output message: %s", message)
    return message


def load_local_messages(*, filename: str | None, logger: Logger) -> list:
    """Load local messages from a text file."""
    return EXAMPLE


def send_message(message: dict, settings: InferenceSettings, logger: Logger):
    """Send message to kafka or write it to file."""
    if settings.LOCAL_MODE:
        if settings.LOCAL_OUT_MESSAGES is None:
            logger.error("LOCAL_OUT_MESSAGES environment variable has not been set.")
            exit(1)
        # 'a' mode appends without overwriting
        with open(settings.LOCAL_OUT_MESSAGES, "a") as file:
            file.write(json.dumps(message, indent=4))
        logger.info("Message written into %s", settings.LOCAL_OUT_MESSAGES)
    else:
        producer = create_kafka_producer(
            kafka_server_url=settings.KAFKA_HOSTNAME, logger=logger
        )
        producer.send(settings.KAFKA_RANKED_PROVIDERS_TOPIC, message)
        producer.close()
        logger.info(
            "Message sent to topic '%s' of kafka server'%s'",
            settings.KAFKA_RANKED_PROVIDERS_TOPIC,
            settings.KAFKA_HOSTNAME,
        )


def run(logger: Logger):
    """Function to load the dataset, do preprocessing, load the
    model from MLFlow and infer the best provider."""
    # Load the settings and setup MLFlow and create the MLflow client
    settings = load_inference_settings(logger=logger)
    setup_mlflow(logger=logger)
    client = MlflowClient()
    if settings.LOCAL_MODE:
        if settings.LOCAL_IN_MESSAGES is None:
            logger.error("LOCAL_IN_MESSAGES environment variable has not been set.")
            exit(1)
        consumer = load_local_messages(
            filename=settings.LOCAL_IN_MESSAGES, logger=logger
        )
    else:
        consumer = create_kafka_consumer(
            kafka_server_url=settings.KAFKA_HOSTNAME,
            topic=settings.KAFKA_INFERENCE_TOPIC,
            partition=settings.KAFKA_INFERENCE_TOPIC_PARTITION,
            offset=settings.KAFKA_INFERENCE_TOPIC_OFFSET,
            logger=logger,
        )

    # Listen for new messages from the inference topic
    for message in consumer:
        logger.info("New message received")
        logger.debug("Message: %s", message)
        if not settings.LOCAL_MODE:
            # Decode bytes to a string
            message = message.value.decode("utf-8")
        data = json.loads(message)
        if len(data["providers"]) > 1:
            logger.info("Select between multiple providers")
            df = load_dataset(settings=settings, logger=logger)
            df = preprocessing(
                df=df,
                complex_templates=settings.TEMPLATE_COMPLEX_TYPES,
                logger=logger,
            )
            input_data = pre_process_message(
                message=data,
                df=df,
                complex_templates=settings.TEMPLATE_COMPLEX_TYPES,
                logger=logger,
            )
            sorted_results = process_inference(
                input_inference=input_data,
                mlflow_client=client,
                settings=settings,
                logger=logger,
            )
        else:
            logger.info("Only one provider. No inference needed")
            el = data["providers"][0]
            provider = f"{el[MSG_PROVIDER_NAME]}-{el[MSG_REGION_NAME]}"
            sorted_results = {provider: 0.5}

        output_message = create_message(
            sorted_results=sorted_results,
            input_data=data["providers"],
            deployment_uuid=data["uuid"],
            logger=logger,
        )
        send_message(output_message, settings=settings, logger=logger)
