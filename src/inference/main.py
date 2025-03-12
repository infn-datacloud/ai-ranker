import ast
import base64
import json
import os
import pickle
import time
from logging import Logger

import mlflow
import mlflow.sklearn
import pandas as pd
from kafka import KafkaConsumer

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
from settings import load_inference_settings, setup_mlflow


def processMessage(
    message: dict, input_list: list, template_complex_types: list, settings, logger
):
    input_message = {}
    exact_flavors_dict = {}
    message_providers = message["providers"]
    template_name = message[MSG_TEMPLATE_NAME]
    complexity = 0
    if template_name in template_complex_types:
        complexity = 1
    df = load_dataset(settings=settings, logger=logger)

    df = preprocessing(
        df=df,
        complex_templates=template_complex_types,
        logger=logger,
    )
    for el in message_providers:
        provider = el[MSG_PROVIDER_NAME] + "-" + el[MSG_REGION_NAME]
        df_filtered = df[
            (df[MSG_TEMPLATE_NAME].isin([template_name]))
            & (df[DF_PROVIDER].isin([provider]))
        ]
        df_filtered = df_filtered.copy()
        avg_success_time = (
            float(
                df_filtered.loc[df_filtered["timestamp"].idxmax(), DF_AVG_SUCCESS_TIME]
            )
            if not df_filtered.empty
            else 0.0
        )
        avg_failure_time = (
            float(df_filtered.loc[df_filtered["timestamp"].idxmax(), DF_AVG_FAIL_TIME])
            if not df_filtered.empty
            else 0.0
        )
        failure_percentage = (
            float(df_filtered.loc[df_filtered["timestamp"].idxmax(), DF_FAIL_PERC])
            if not df_filtered.empty
            else 0.0
        )
        calculated_values = {
            DF_CPU_DIFF: (el[MSG_CPU_QUOTA] - el[MSG_CPU_REQ]) - el[MSG_CPU_USAGE],
            DF_RAM_DIFF: (el[MSG_RAM_QUOTA] - el[MSG_RAM_REQ]) - el[MSG_RAM_USAGE],
            DF_DISK_DIFF: (el[MSG_DISK_QUOTA] - el[MSG_DISK_REQ]) - el[MSG_DISK_USAGE],
            DF_INSTANCE_DIFF: (el[MSG_INSTANCE_QUOTA] - el[MSG_INSTANCE_REQ])
            - el[MSG_INSTANCE_USAGE],
            DF_PUB_IPS_DIFF: (el[MSG_PUB_IPS_QUOTA] - el[MSG_PUB_IPS_REQ])
            - el[MSG_PUB_IPS_USAGE],
            DF_GPU: float(bool(el[MSG_GPU_REQ])),
            "test_failure_perc_30d": el["test_failure_perc_30d"],
            "test_failure_perc_7d": el["test_failure_perc_7d"],
            "test_failure_perc_1d": el["test_failure_perc_1d"],
            DF_COMPLEX: complexity,
            "overbooking_ram": el["overbooking_ram"],
            "overbooking_cpu": el["overbooking_cpu"],
            DF_AVG_SUCCESS_TIME: avg_success_time,
            DF_AVG_FAIL_TIME: avg_failure_time,
            DF_FAIL_PERC: failure_percentage,
        }
        exact_flavors_dict[provider] = 1.0 - float(
            bool(el[MSG_INSTANCE_REQ] - el["exact_flavors"])
        )
        input_message[provider] = [
            calculated_values[key] for key in input_list if key in calculated_values
        ]
    return input_message, exact_flavors_dict


def classification_predict(
    inputData: dict,
    feature_input: list,
    model_name: str,
    model_version: str,
    scaling_enable: bool,
    scaler_file: str,
):
    model_uri = f"models:/{model_name}/{model_version}"
    classification_values = {}
    try:
        # Get the model type and load it with the proper function
        model = mlflow.pyfunc.load_model(model_uri)
        model_type = model.metadata.flavors.keys()
        if "sklearn" in model_type:
            model = mlflow.sklearn.load_model(model_uri)
        else:
            raise ValueError("Model type not supported")

        # Load model scaler if scaling is enabled
        if scaling_enable:
            scaler_uri = f"{model_uri}/{scaler_file}"
            scaler_dict = mlflow.artifacts.load_dict(scaler_uri)
            scaler_bytes = base64.b64decode(scaler_dict["scaler"])
            scaler = pickle.loads(scaler_bytes)
        for key, data in inputData.items():
            X_new = pd.DataFrame([data], columns=feature_input)
            # scale x_new if scaling is enabled
            if scaling_enable:
                X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns, index=X_new.index)
            else:
                X_new_scaled = X_new
            y_pred_new = model.predict_proba(X_new_scaled)
            classification_response = y_pred_new.tolist()
            success_prob = classification_response[0][0]
            classification_values[key] = success_prob
        return classification_values

    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")


def regression_predict(
    inputData: dict,
    feature_input: list,
    model_name: str,
    model_version: str,
    max_regression_time,
    min_regression_time,
    scaling_enable: str,
    scaler_file: str,
):
    model_uri = f"models:/{model_name}/{model_version}"
    regression_values = {}
    try:
        # Get the model type and load it with the proper function
        model = mlflow.pyfunc.load_model(model_uri)
        model_type = model.metadata.flavors.keys()

        if "sklearn" in model_type:
            model = mlflow.sklearn.load_model(model_uri)

        else:
            raise ValueError("Model type not supported")

        # Load model scaler if scaling is enabled
        if scaling_enable:
            scaler_uri = f"{model_uri}/{scaler_file}"
            scaler_dict = mlflow.artifacts.load_dict(scaler_uri)
            scaler_bytes = base64.b64decode(scaler_dict["scaler"])
            scaler = pickle.loads(scaler_bytes)
        for key, data in inputData.items():
            X_new = pd.DataFrame([data], columns=feature_input)
            # scale x_new if scaling is enabled
            if scaling_enable:
                X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns, index=X_new.index)
            else:
                X_new_scaled = X_new
            y_pred_new = model.predict(X_new_scaled)
            regression_response = y_pred_new.tolist()
            regression_value_raw = regression_response[0]
            regression_value = 1 - (regression_value_raw - min_regression_time) / (
                max_regression_time - min_regression_time
            )
            regression_values[key] = regression_value
        return regression_values

    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")


def process_inference(
    input_inference: dict,
    feature_input: list,
    exact_flavors_dict: dict,
    classification_model_name: str,
    classification_model_version: str,
    regression_model_name: str,
    regression_model_version: str,
    min_regression_time: int,
    max_regression_time: int,
    exact_flavour: bool,
    scaling_enable: bool,
    scaler_file: str,
    classification_weight: float = 0.75,
    threshold: float = 0.7,
    filter_on: bool = False,
) -> dict:
    results = {}
    ##### DEFAULT VALUE
    default_value = 0.1

    try:
        # Send requests to classification and regression endpoints
        classification_response = classification_predict(
            inputData=input_inference,
            feature_input=feature_input,
            model_name=classification_model_name,
            model_version=classification_model_version,
            scaling_enable=scaling_enable,
            scaler_file=scaler_file,
        )
        regression_response = regression_predict(
            inputData=input_inference,
            feature_input=feature_input,
            model_name=regression_model_name,
            model_version=regression_model_version,
            min_regression_time=min_regression_time,
            max_regression_time=max_regression_time,
            scaling_enable=scaling_enable,
            scaler_file=scaler_file,
        )

        for key, value in classification_response.items():
            results[key] = value * classification_weight + regression_response[key] * (
                1 - classification_weight
            )

    except (KeyError, IndexError, TypeError) as e:
        print(f"Unexpected response structure for {key}: {e}")
        ##### DEFAULT VALUE
        results[key] = default_value  # Default value for unexpected responses
    sorted_results = dict(
        sorted(results.items(), key=lambda item: item[1], reverse=True)
    )
    # print("prima:\n ",json.dumps(sorted_results, indent=4))
    # print(json.dumps(input_inference, indent=4))
    # Filter results based on the threshold
    if exact_flavour:
        exact = [k for k in input_inference if exact_flavors_dict[k] == 1.0]
        no_exact = [k for k in input_inference if exact_flavors_dict[k] == 0.0]

        exact_sorted = sorted(exact, key=lambda k: sorted_results[k], reverse=True)
        no_exact_sorted = sorted(
            no_exact, key=lambda k: sorted_results[k], reverse=True
        )

        sorted_keys = exact_sorted + no_exact_sorted

        sorted_results = {k: sorted_results[k] for k in sorted_keys}
        # print("dopo:\n ",json.dumps(sorted_results, indent=4))
    if filter_on:
        sorted_results = {
            key: value for key, value in results.items() if value >= threshold
        }

    return sorted_results


def get_latest_version(client, model_name: str) -> str:
    latest_version = max(
        client.search_model_versions(f"name='{model_name}'"),
        key=lambda x: int(x.version),
    ).version
    return latest_version


def get_features_input(client, model_name: str) -> list:
    latest_version = get_latest_version(client, model_name)
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.sklearn.load_model(model_uri)
    feature_names = list(model.feature_names_in_)
    return feature_names


def create_message(sorted_results: dict, deployment_uuid: str) -> str:
    ranked_providers = [
        {MSG_PROVIDER_NAME: provider, "value": value}
        for provider, value in sorted_results.items()
    ]
    message = {"uuid": deployment_uuid, "ranked_providers": ranked_providers}
    return json.dumps(message, indent=4)


def run(logger: Logger):
    settings = load_inference_settings()
    setup_mlflow(logger=logger)

    classification_model_name = settings.CLASSIFICATION_MODEL_NAME
    classification_model_version = settings.CLASSIFICATION_MODEL_VERSION
    regression_model_name = settings.REGRESSION_MODEL_NAME
    regression_model_version = settings.REGRESSION_MODEL_VERSION
    min_regression_time = settings.REGRESSION_MIN_TIME
    max_regression_time = settings.REGRESSION_MAX_TIME
    filter_mode = settings.FILTER
    threshold = settings.THRESHOLD
    classification_weight = settings.CLASSIFICATION_WEIGHT
    exact_flavour = settings.EXACT_FLAVOUR_PRECEDENCE
    scaler_file = settings.SCALER_FILE
    scaling_enable = settings.SCALING_ENABLE

    # Create the MLflow client
    client = mlflow.tracking.MlflowClient()
    if classification_model_version == "latest":
        classification_model_version = get_latest_version(
            client, classification_model_name
        )
    if regression_model_version == "latest":
        regression_model_version = get_latest_version(client, regression_model_name)

    feature_input = get_features_input(client, classification_model_name)

    #### KAFKA SETUP

    kafka_server_url = os.environ.get("KAFKA_HOSTNAME", settings.KAFKA_URL)
    topic = os.environ.get("KAFKA_TOPIC", "test")
    template_complex_types = settings.TEMPLATE_COMPLEX_TYPES
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=kafka_server_url,
        # auto_offset_reset='earliest'
    )
    with open("output_messages.txt", "w") as file:
        pass

    if consumer.bootstrap_connected():
        print("Connected")
        print(f"Subscribed topics: {consumer.subscription()}")

        for message in consumer:
            # print(message)
            message = message.value.decode("utf-8")  # Decode bytes to a string
            message = json.loads(message)
            deployment_uuid = message["uuid"]
            if len(message["providers"]) != 1:
                try:
                    input_dict, exact_flavors_dict = processMessage(
                        message, feature_input, template_complex_types, settings, logger
                    )
                    start_time = time.time()
                    sorted_results = process_inference(
                        input_dict,
                        feature_input,
                        exact_flavors_dict,
                        classification_model_name,
                        classification_model_version,
                        regression_model_name,
                        regression_model_version,
                        min_regression_time=min_regression_time,
                        max_regression_time=max_regression_time,
                        exact_flavour=exact_flavour,
                        classification_weight=classification_weight,
                        threshold=threshold,
                        filter_on=filter_mode,
                        scaling_enable=scaling_enable,
                        scaler_file=scaler_file,
                    )
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Inference Time: {elapsed_time:.2f} seconds")
                except Exception as e:
                    # Handle exceptions and log them
                    print(f"Error processing message: {e}")
                    sorted_results = {}
                    for el in message["providers"]:
                        sorted_results[
                            el[MSG_PROVIDER_NAME] + "-" + el[MSG_REGION_NAME]
                        ] = 0.5
            else:
                el = message["providers"][0]
                provider = el[MSG_PROVIDER_NAME] + "-" + el[MSG_REGION_NAME]
                sorted_results = {provider: 0.5}
            output_message = create_message(sorted_results, deployment_uuid)
            with open(
                "output_messages.txt", "a"
            ) as file:  # 'a' mode appends without overwriting
                file.write(output_message)
