import ast
import json
import os
import time
from logging import Logger

import mlflow
import mlflow.sklearn
import pandas as pd
from kafka import KafkaConsumer

from processing import load_dataset_from_kafka, preprocessing
from settings import load_inference_settings, setup_mlflow


def processMessage(message: dict, input_list: list, template_complex_types: list):
    input_message = {}
    exact_flavors_dict = {}
    message_providers = message["providers"]
    template_name = message["template_name"]
    complexity = 0
    if template_name in template_complex_types:
        complexity = 1
    df = load_dataset_from_kafka(
        kafka_server_url="localhost:9092", topic="training", partition=0, offset=765
    )

    df = preprocessing(df, template_complex_types)
    for el in message_providers:
        provider = el["provider_name"] + "-" + el["region_name"]
        df_filtered = df[
            (df["template_name"].isin([template_name]))
            & (df["provider"].isin([provider]))
        ]
        df_filtered = df_filtered.copy()
        avg_success_time = (
            float(
                df_filtered.loc[df_filtered["timestamp"].idxmax(), "avg_success_time"]
            )
            if not df_filtered.empty
            else 0.0
        )
        avg_failure_time = (
            float(
                df_filtered.loc[df_filtered["timestamp"].idxmax(), "avg_failure_time"]
            )
            if not df_filtered.empty
            else 0.0
        )
        failure_percentage = (
            float(
                df_filtered.loc[df_filtered["timestamp"].idxmax(), "failure_percentage"]
            )
            if not df_filtered.empty
            else 0.0
        )
        calculated_values = {
            "cpu_diff": (el["vcpus_quota"] - el["vcpus_requ"]) - el["vcpus_usage"],
            "ram_diff": (el["ram_gb_quota"] - el["ram_gb_requ"]) - el["ram_gb_usage"],
            "storage_diff": (el["storage_gb_quota"] - el["storage_gb_requ"])
            - el["storage_gb_usage"],
            "instances_diff": (el["n_instances_quota"] - el["n_instances_requ"])
            - el["n_instances_usage"],
            "floatingips_diff": (el["floating_ips_quota"] - el["floating_ips_requ"])
            - el["floating_ips_usage"],
            "gpu": float(bool(el["gpus_requ"])),
            "test_failure_perc_30d": el["test_failure_perc_30d"],
            "test_failure_perc_7d": el["test_failure_perc_7d"],
            "test_failure_perc_1d": el["test_failure_perc_1d"],
            "complexity": complexity,
            "overbooking_ram": el["overbooking_ram"],
            "overbooking_cpu": el["overbooking_cpu"],
            "avg_success_time": avg_success_time,
            "avg_failure_time": avg_failure_time,
            "failure_percentage": failure_percentage,
        }
        exact_flavors_dict[provider] = 1.0 - float(
            bool(el["n_instances_requ"] - el["exact_flavors"])
        )
        input_message[provider] = [
            calculated_values[key] for key in input_list if key in calculated_values
        ]
    return input_message, exact_flavors_dict


def classification_predict(
    inputData: dict, feature_input: list, model_name: str, model_version: str
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
        for key, data in inputData.items():
            X_new = pd.DataFrame([data], columns=feature_input)
            y_pred_new = model.predict_proba(X_new)
            classification_response = y_pred_new.tolist()
            success_prob = classification_response[0][0]
            classification_values[key] = success_prob
        return classification_values

    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")


def regression_predict(
    inputData: dict,
    model_name: str,
    model_version: str,
    max_regression_time,
    min_regression_time,
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
        for key, data in inputData.items():
            X_new = pd.DataFrame([data])  # columns = feature_names)
            y_pred_new = model.predict(X_new)
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
        )
        regression_response = regression_predict(
            inputData=input_inference,
            model_name=regression_model_name,
            model_version=regression_model_version,
            min_regression_time=min_regression_time,
            max_regression_time=max_regression_time,
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
    latestVersion = client.get_latest_versions(model_name)
    return latestVersion[0].version


def get_features_input(client, model_name: str) -> list:
    latestVersion = client.get_latest_versions(model_name)
    run_id = latestVersion[0].run_id
    run = mlflow.get_run(run_id)
    feature_names = run.data.tags.get("features", None)
    return feature_names


def create_message(sorted_results: dict, deployment_uuid: str) -> str:
    ranked_providers = [
        {"provider_name": provider, "value": value}
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

    # Create the MLflow client
    client = mlflow.tracking.MlflowClient()
    if classification_model_version == "latest":
        classification_model_version = get_latest_version(
            client, classification_model_name
        )
    if regression_model_version == "latest":
        regression_model_version = get_latest_version(client, regression_model_name)

    feature_names = get_features_input(client, classification_model_name)
    feature_input = ast.literal_eval(feature_names)
    feature_input = feature_input[:-2]

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
                        message, feature_input, template_complex_types
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
                            el["provider_name"] + "-" + el["region_name"]
                        ] = 0.5
            else:
                el = message["providers"][0]
                provider = el["provider_name"] + "-" + el["region_name"]
                sorted_results = {provider: 0.5}
            output_message = create_message(sorted_results, deployment_uuid)
            with open(
                "output_messages.txt", "a"
            ) as file:  # 'a' mode appends without overwriting
                file.write(output_message)
