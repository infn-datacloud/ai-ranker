import json
import time
from logging import Logger
from typing import Any

import pandas as pd
from mlflow import MlflowClient
from sklearn.preprocessing import RobustScaler

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
    load_training_data,
    preprocessing,
)
from settings import InferenceSettings, load_inference_settings
from utils.kafka import create_kafka_consumer, create_kafka_producer
from utils.mlflow import get_model, get_model_uri, get_scaler, setup_mlflow

CLASSIFICATION_SUCCESS_IDX = 0
NO_PREDICTED_VALUE = -1
K_RES_EXACT = "resource_exactness"
K_CLASS = "classification"
K_REGR = "regression"


def create_inference_input(
    *, data: dict, model: Any, scaler: RobustScaler | None
) -> pd.DataFrame:
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
        x_new = create_inference_input(data=data, model=sklearn_model, scaler=scaler)
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
        x_new = create_inference_input(data=data, model=sklearn_model, scaler=scaler)
        logger.info("Predict time for '%s'", provider)
        y_pred_new = sklearn_model.predict(x_new)
        expected_time = float(y_pred_new[0])
        # Convert this value into a value between (0,1) range.
        # If expected_time < min_regression_time this value is a very good value.
        if max_regression_time > 0:
            regression_value = 1 - (expected_time - min_regression_time) / (
                max_regression_time - min_regression_time
            )
        else:
            regression_value = 1

        logger.debug("Predicted time for '%s': %.2f", provider, expected_time)
        logger.debug("Predicted goodness for '%s': %.2f", provider, expected_time)
        regression_values[provider] = regression_value
    return regression_values


def predict(
    *,
    input_inference: dict,
    mlflow_client: MlflowClient,
    settings: InferenceSettings,
    logger: Logger,
) -> tuple[dict, dict]:
    """Predict classification and regression on all input data"""
    start_time = time.time()

    # Send requests to classification and regression endpoints
    try:
        classification_response = classification_predict(
            input_data=input_inference,
            model_name=settings.CLASSIFICATION_MODEL_NAME,
            model_version=settings.CLASSIFICATION_MODEL_VERSION,
            scaler_file=settings.SCALER_FILE if settings.SCALING_ENABLE else None,
            mlflow_client=mlflow_client,
            logger=logger,
        )
    except ValueError as e:
        logger.error(e)
        classification_response = {
            k: NO_PREDICTED_VALUE for k in input_inference.keys()
        }
    try:
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
    except ValueError as e:
        logger.error(e)
        regression_response = {k: NO_PREDICTED_VALUE for k in input_inference.keys()}

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug("Inference Time: %.2f seconds", elapsed_time)

    assert len(classification_response) == len(regression_response), (
        "Responses length mismatch."
    )
    return classification_response, regression_response


def sort_key_with_exact_res(item: dict) -> tuple[float, float]:
    """Sort results by resource exactness and then success probability"""
    return item[1][K_RES_EXACT], item[1][K_CLASS] + item[1][K_REGR]


def sort_key(item: dict) -> float:
    """Sort results by success probability"""
    return item[1][K_CLASS] + item[1][K_REGR]


def merge_and_sort_results(
    *,
    input_inference: dict,
    classification_response: dict,
    regression_response: dict,
    settings: InferenceSettings,
    logger: Logger,
):
    """Merge classification and regression results to sort providers."""
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
        sort_func = sort_key_with_exact_res
    else:
        sort_func = sort_key
    sorted_results = dict(sorted(results.items(), key=sort_func, reverse=True))
    logger.debug("Sorted results: %s", results)
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

        # Retrieve latest details for the combination: provider-region-template
        avg_success_time = 0.0
        avg_failure_time = 0.0
        failure_percentage = 0.0
        if not df.empty:
            df = df[
                (df[MSG_TEMPLATE_NAME] == template_name)
                & (df[DF_PROVIDER] == provider_region)
            ]
        if not df.empty:
            idx_latest = df[DF_TIMESTAMP].idxmax()
            avg_success_time = float(df.loc[idx_latest, DF_AVG_SUCCESS_TIME])
            avg_failure_time = float(df.loc[idx_latest, DF_AVG_FAIL_TIME])
            failure_percentage = float(df.loc[idx_latest, DF_FAIL_PERC])

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


def load_local_messages(*, filename: str | None, logger: Logger) -> list[dict]:
    """Load local messages from a text file."""
    with open(filename) as file:
        messages = json.load(file)
    logger.debug("Loaded messages: %s", messages)
    return messages


def send_message(message: dict, settings: InferenceSettings, logger: Logger) -> None:
    """Send message to kafka or write it to file."""
    if settings.LOCAL_MODE:
        if settings.LOCAL_OUT_MESSAGES is None:
            logger.error("LOCAL_OUT_MESSAGES environment variable has not been set.")
            return
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


def run(logger: Logger) -> None:
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
            consumer_timeout_ms=settings.KAFKA_INFERENCE_TOPIC_TIMEOUT,
            logger=logger,
        )

    # Listen for new messages from the inference topic
    for message in consumer:
        logger.info("New message received")
        if not settings.LOCAL_MODE:
            logger.debug("Message: %s", message)
            data = message.value
        else:
            data = message
        logger.debug("Message data: %s", data)

        if len(data["providers"]) > 1:
            logger.info("Select between multiple providers")
            df = load_training_data(settings=settings, logger=logger)
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
            classification_prediction, regression_prediction = predict(
                input_inference=input_data,
                mlflow_client=client,
                settings=settings,
                logger=logger,
            )
            sorted_results = merge_and_sort_results(
                input_inference=input_data,
                classification_response=classification_prediction,
                regression_response=regression_prediction,
                settings=settings,
                logger=logger,
            )
        else:
            logger.info("Only one provider. No inference needed")
            el = data["providers"][0]
            provider = f"{el[MSG_PROVIDER_NAME]}-{el[MSG_REGION_NAME]}"
            sorted_results = {
                provider: {
                    K_CLASS: NO_PREDICTED_VALUE,
                    K_REGR: NO_PREDICTED_VALUE,
                    K_RES_EXACT: el[MSG_INSTANCES_WITH_EXACT_FLAVORS]
                    / el[MSG_INSTANCE_REQ],
                    **el,
                }
            }

        output_message = create_message(
            sorted_results=sorted_results,
            input_data=data["providers"],
            deployment_uuid=data["uuid"],
            logger=logger,
        )
        send_message(output_message, settings=settings, logger=logger)
