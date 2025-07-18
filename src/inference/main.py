import time
from logging import Logger
from typing import Any

import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from mlflow import MlflowClient
from sklearn.preprocessing import RobustScaler

from src.exceptions import ConfigurationError
from src.processing import (
    DF_AVG_FAIL_TIME,
    DF_AVG_SUCCESS_TIME,
    DF_FAIL_PERC,
    DF_MAX_DEP_TIME,
    DF_MIN_DEP_TIME,
    DF_PROVIDER,
    DF_TIMESTAMP,
    calculate_derived_properties,
    preprocessing,
)
from src.settings import InferenceSettings, TrainingSettings, load_inference_settings
from src.training.main import load_training_data
from src.utils import (
    MSG_DEP_UUID,
    MSG_INSTANCE_REQ,
    MSG_INSTANCES_WITH_EXACT_FLAVORS,
    MSG_PROVIDER_NAME,
    MSG_REGION_NAME,
    MSG_TEMPLATE_NAME,
    MSG_VALID_KEYS,
    MSG_VERSION,
    load_data_from_file,
    write_data_to_file,
)
from src.utils.kafka import create_kafka_consumer, create_kafka_producer
from src.utils.mlflow import get_model, get_model_uri, get_scaler, setup_mlflow

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
    values = [[data.get(k, np.nan) for k in input_features]]
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
    if mlflow_model.metadata.metadata["scaling"]:
        logger.info("Load scaler from MLFlow.")
        scaler = get_scaler(
            model_uri=model_uri,
            scaler_file=mlflow_model.metadata.metadata["scaler_file"],
        )

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
    if mlflow_model.metadata.metadata["scaling"]:
        logger.info("Load scaler from MLFlow.")
        scaler = get_scaler(
            model_uri=model_uri,
            scaler_file=mlflow_model.metadata.metadata["scaler_file"],
        )

    # Calculate expected time for each provider
    regression_values = {}
    for provider, data in input_data.items():
        x_new = create_inference_input(data=data, model=sklearn_model, scaler=scaler)
        logger.info("Predict time for '%s'", provider)
        y_pred_new = sklearn_model.predict(x_new)
        expected_time = float(y_pred_new[0])
        # Convert this value into a value between (0,1) range.
        # If expected_time < min_regression_time this value is a very good value.
        min_regression_time = input_data[provider][DF_MIN_DEP_TIME]
        max_regression_time = input_data[provider][DF_MAX_DEP_TIME]
        if max_regression_time > 0:
            raw_val = 1 - (expected_time - min_regression_time) / (
                max_regression_time - min_regression_time
            )
            regression_value = float(np.clip(raw_val, 0, 1))
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
        logger.info("Start classification prediction.")
        classification_response = classification_predict(
            input_data=input_inference,
            model_name=settings.CLASSIFICATION_MODEL_NAME,
            model_version=settings.CLASSIFICATION_MODEL_VERSION,
            mlflow_client=mlflow_client,
            logger=logger,
        )
    except ValueError as e:
        logger.error(e)
        classification_response = {
            k: NO_PREDICTED_VALUE for k in input_inference.keys()
        }
    try:
        logger.info("Start regression prediction.")
        regression_response = regression_predict(
            input_data=input_inference,
            model_name=settings.REGRESSION_MODEL_NAME,
            model_version=settings.REGRESSION_MODEL_VERSION,
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
    msg_version = message.pop(MSG_VERSION)
    msg_map = MSG_VALID_KEYS.get(msg_version, None)
    if msg_map is None:
        raise ValueError(f"Message version {msg_version} not supported")

    for request in message["providers"]:
        invalid_keys = set(request.keys()).difference(msg_map)
        assert len(invalid_keys) == 0, f"Found invalid keys: {invalid_keys}"

    df_msg = pd.DataFrame(message["providers"])
    df_msg[MSG_TEMPLATE_NAME] = message[MSG_TEMPLATE_NAME]
    df_msg = calculate_derived_properties(
        df=df_msg, complex_templates=complex_templates
    )
    for _, row in df_msg.iterrows():
        # Filter historical values related to the target provider and region
        # and retrieve latest details about success and failure percentage.
        avg_success_time = 0.0
        avg_failure_time = 0.0
        failure_percentage = 0.0
        min_time = 0.0
        max_time = 0.0
        if not df.empty:
            df = df[
                (df[MSG_TEMPLATE_NAME] == row[MSG_TEMPLATE_NAME])
                & (df[DF_PROVIDER] == row[DF_PROVIDER])
            ]
        if not df.empty:
            idx_latest = df[DF_TIMESTAMP].idxmax()
            avg_success_time = float(df.loc[idx_latest, DF_AVG_SUCCESS_TIME])
            avg_failure_time = float(df.loc[idx_latest, DF_AVG_FAIL_TIME])
            failure_percentage = float(df.loc[idx_latest, DF_FAIL_PERC])
            min_time = float(df.loc[idx_latest, DF_MIN_DEP_TIME])
            max_time = float(df.loc[idx_latest, DF_MAX_DEP_TIME])
        else:
            logger.info("No historical data about this kind of requests")

        data[row[DF_PROVIDER]] = {
            **row.to_dict(),
            DF_AVG_SUCCESS_TIME: avg_success_time,
            DF_AVG_FAIL_TIME: avg_failure_time,
            DF_FAIL_PERC: failure_percentage,
            K_RES_EXACT: row[MSG_INSTANCES_WITH_EXACT_FLAVORS] / row[MSG_INSTANCE_REQ]
            if row[MSG_INSTANCE_REQ] > 0
            else 0,
            DF_MAX_DEP_TIME: max_time,
            DF_MIN_DEP_TIME: min_time,
        }
        logger.debug(
            "The following data has been requested on provider '%s' on region '%s': %s",
            row[MSG_PROVIDER_NAME],
            row[MSG_REGION_NAME],
            data[row[DF_PROVIDER]],
        )

    return data


def create_message(
    *, sorted_results: dict, input_data: dict, deployment_uuid: str, logger: Logger
) -> dict[str, Any]:
    """Create a dict with the deployment uuid and the list of ranked providers."""
    ranked_providers = []
    for provider, values in sorted_results.items():
        data = None
        for i in input_data:
            if f"{i[MSG_PROVIDER_NAME]}-{i[MSG_REGION_NAME]}" == provider:
                data = {**values, **i}
                break
        if data is None:
            msg = f"No matching input_data entry for provider key '{provider}'"
            logger.error(msg)
            raise ValueError(msg)
        ranked_providers.append(data)
    message = {MSG_DEP_UUID: deployment_uuid, "ranked_providers": ranked_providers}
    logger.debug("Output message: %s", message)
    return message


def send_message(message: dict, settings: InferenceSettings, logger: Logger) -> None:
    """Send a message either to a Kafka topic or writes it to a local file.

    Depends on the settings. If `settings.LOCAL_MODE` is True, writes the message to a
    local file specified by `settings.LOCAL_OUT_MESSAGES`. Otherwise, sends the message
    to the Kafka topic specified by `settings.KAFKA_RANKED_PROVIDERS_TOPIC` on the Kafka
    server at `settings.KAFKA_HOSTNAME`. Logs the outcome of the operation.

    Args:
        message (dict): The message payload to be sent or written.
        settings (InferenceSettings): Configuration settings that determine the output
            mode and relevant parameters.
        logger (Logger): Logger instance for logging information about the operation.

    Raises:
        ConfigurationError: generated by either `write_data_to_file` or
            `create_kafka_producer`.

    """
    """Send message to kafka or write it to file."""
    if settings.LOCAL_MODE:
        write_data_to_file(filename=settings.LOCAL_OUT_MESSAGES, data=message)
        logger.info("Message written into '%s'", settings.LOCAL_OUT_MESSAGES)
        return

    producer = create_kafka_producer(settings=settings, logger=logger)
    producer.send(settings.KAFKA_RANKED_PROVIDERS_TOPIC, message)
    producer.close()
    logger.info(
        "Message sent to topic '%s' of kafka server'%s'",
        settings.KAFKA_RANKED_PROVIDERS_TOPIC,
        settings.KAFKA_HOSTNAME,
    )


def connect_consumers_or_load_data(
    settings: InferenceSettings, logger: Logger
) -> tuple[KafkaConsumer, KafkaConsumer] | tuple[list[dict], list[dict]]:
    """Initialize Kafka consumers or loads data from files based on provided settings.

    If `settings.LOCAL_MODE` is True, loads input and output messages from local files
    specified in the settings. Otherwise, creates and returns Kafka consumers for input
    and output topics.

    Args:
        settings (InferenceSettings): Configuration settings for inference, including
            Kafka and file paths.
        logger (Logger): Logger instance for logging messages and errors.

    Returns:
        tuple[KafkaConsumer, KafkaConsumer] | tuple[list[dict], list[dict]]:
            - If in local mode, returns two lists of dictionaries representing input and
                output messages.
            - If not in local mode, returns two KafkaConsumer instances for input and
                output topics.

    Raises:
        ConfigurationError: generated by either `load_data_from_file` or
            `create_kafka_consumer`.
    """
    if settings.LOCAL_MODE:
        inputs = load_data_from_file(filename=settings.LOCAL_IN_MESSAGES, logger=logger)
        outputs = load_data_from_file(
            filename=settings.LOCAL_OUT_MESSAGES, logger=logger
        )
        return inputs, outputs

    input_consumer = create_kafka_consumer(
        settings=settings,
        topic=settings.KAFKA_INFERENCE_TOPIC,
        client_id=settings.KAFKA_INFERENCE_CLIENT_NAME,
        consumer_timeout_ms=settings.KAFKA_INFERENCE_TOPIC_TIMEOUT,
        logger=logger,
    )
    output_consumer = create_kafka_consumer(
        settings=settings,
        topic=settings.KAFKA_RANKED_PROVIDERS_TOPIC,
        client_id=settings.KAFKA_INFERENCE_CLIENT_NAME,
        consumer_timeout_ms=settings.KAFKA_RANKED_PROVIDERS_TOPIC_TIMEOUT,
        logger=logger,
    )
    return input_consumer, output_consumer


def run(logger: Logger) -> None:
    """Function to load the dataset, do preprocessing, load the
    model from MLFlow and infer the best provider."""
    # Load the settings and setup MLFlow and create the MLflow client
    settings = load_inference_settings(logger=logger)
    client = setup_mlflow(logger=logger)

    try:
        input_consumer, output_consumer = connect_consumers_or_load_data(
            settings=settings, logger=logger
        )
    except ConfigurationError as e:
        logger.error(e.message)
        exit(1)

    if not settings.LOCAL_MODE:
        processed_dep_uuids = [
            message.value[MSG_DEP_UUID] for message in output_consumer
        ]
    else:
        processed_dep_uuids = [message[MSG_DEP_UUID] for message in output_consumer]

    # Listen for new messages from the inference topic
    logger.info("Start listening for new messages")
    for message in input_consumer:
        aborted = False

        logger.info("New message received")
        if not settings.LOCAL_MODE:
            logger.debug("Message: %s", message)
            data = message.value
        else:
            data = message
        logger.debug("Message data: %s", data)

        # Skip already processed messages
        idx = -1
        try:
            if len(processed_dep_uuids) > 0:
                idx = processed_dep_uuids.index(data[MSG_DEP_UUID])
        except ValueError:
            pass
        if idx != -1:
            logger.info("Already processed message. Skipping")
            _ = processed_dep_uuids.pop(idx)
            continue

        # Process new message.
        if len(data["providers"]) > 1:
            logger.info("Select between multiple providers")
            try:
                train_settings = TrainingSettings(**settings.model_dump())
                train_settings.KAFKA_TRAINING_CLIENT_NAME = (
                    settings.KAFKA_INFERENCE_CLIENT_NAME
                )
                df = load_training_data(settings=train_settings, logger=logger)
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
            except ConfigurationError as e:
                logger.error(e.message)
                aborted = True
        elif len(data["providers"]) == 1:
            logger.info("Only one provider. No inference needed")
            el = data["providers"][0]
            provider = f"{el[MSG_PROVIDER_NAME]}-{el[MSG_REGION_NAME]}"
            if el[MSG_INSTANCE_REQ] > 0:
                res_exact = el[MSG_INSTANCES_WITH_EXACT_FLAVORS] / el[MSG_INSTANCE_REQ]
            else:
                res_exact = 0
            sorted_results = {
                provider: {
                    K_CLASS: NO_PREDICTED_VALUE,
                    K_REGR: NO_PREDICTED_VALUE,
                    K_RES_EXACT: res_exact,
                    **el,
                }
            }
        else:
            aborted = True
            logger.info("No 'providers' available for this request")

        if aborted:
            logger.error("Inference process aborted")
        else:
            output_message = create_message(
                sorted_results=sorted_results,
                input_data=data["providers"],
                deployment_uuid=data["uuid"],
                logger=logger,
            )
            try:
                send_message(output_message, settings=settings, logger=logger)
            except ConfigurationError as e:
                logger.error(e.message)
