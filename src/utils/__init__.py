import json
import os
from logging import Logger
from typing import Any

import pandas as pd
from kafka import KafkaConsumer

from src.exceptions import ConfigurationError

MSG_CPU_QUOTA = "vcpus_quota"
MSG_CPU_USAGE = "vcpus_usage"
MSG_CPU_REQ = "vcpus_requ"
MSG_RAM_QUOTA = "ram_gb_quota"
MSG_RAM_USAGE = "ram_gb_usage"
MSG_RAM_REQ = "ram_gb_requ"
MSG_DISK_QUOTA = "storage_gb_quota"
MSG_DISK_USAGE = "storage_gb_usage"
MSG_DISK_REQ = "storage_gb_requ"
MSG_INSTANCE_QUOTA = "n_instances_quota"
MSG_INSTANCE_USAGE = "n_instances_usage"
MSG_INSTANCE_REQ = "n_instances_requ"
MSG_VOL_QUOTA = "n_volumes_quota"
MSG_VOL_USAGE = "n_volumes_usage"
MSG_VOL_REQ = "n_volumes_requ"
MSG_PUB_IPS_QUOTA = "floating_ips_quota"
MSG_PUB_IPS_USAGE = "floating_ips_usage"
MSG_PUB_IPS_REQ = "floating_ips_requ"
MSG_GPU_REQ = "gpus_requ"
MSG_STATUS = "status"
MSG_STATUS_REASON = "status_reason"
MSG_TEMPLATE_NAME = "template_name"
MSG_DEP_COMPLETION_TIME = "completion_time_s"
MSG_DEP_FAILED_TIME = "tot_failure_time_s"
MSG_DEP_TOT_FAILURES = "n_failures"
MSG_PROVIDER_NAME = "provider_name"
MSG_REGION_NAME = "region_name"
MSG_TIMESTAMP = "submission_time"
MSG_INSTANCES_WITH_EXACT_FLAVORS = "exact_flavors"
MSG_DEP_UUID = "uuid"
MSG_TEST_FAIL_PERC_30D = "test_failure_perc_30d"
MSG_TEST_FAIL_PERC_7D = "test_failure_perc_7d"
MSG_TEST_FAIL_PERC_1D = "test_failure_perc_1d"
MSG_OVERBOOK_RAM = "overbooking_ram"
MSG_OVERBOOK_CORES = "overbooking_cpu"
MSG_BANDWIDTH_IN = "bandwidth_in"
MSG_BANDWIDTH_OUT = "bandwidth_out"
MSG_USER_GROUP = "user_group"
MSG_VERSION = "msg_version"

MSG_VALID_KEYS = {
    "1.1.0": [
        MSG_CPU_QUOTA,
        MSG_CPU_USAGE,
        MSG_CPU_REQ,
        MSG_RAM_QUOTA,
        MSG_RAM_USAGE,
        MSG_RAM_REQ,
        MSG_DISK_QUOTA,
        MSG_DISK_USAGE,
        MSG_DISK_REQ,
        MSG_INSTANCE_QUOTA,
        MSG_INSTANCE_USAGE,
        MSG_INSTANCE_REQ,
        MSG_VOL_QUOTA,
        MSG_VOL_USAGE,
        MSG_VOL_REQ,
        MSG_PUB_IPS_QUOTA,
        MSG_PUB_IPS_USAGE,
        MSG_PUB_IPS_REQ,
        MSG_GPU_REQ,
        MSG_STATUS,
        MSG_STATUS_REASON,
        MSG_TEMPLATE_NAME,
        MSG_DEP_COMPLETION_TIME,
        MSG_DEP_FAILED_TIME,
        MSG_DEP_TOT_FAILURES,
        MSG_PROVIDER_NAME,
        MSG_REGION_NAME,
        MSG_TIMESTAMP,
        MSG_INSTANCES_WITH_EXACT_FLAVORS,
        MSG_DEP_UUID,
        MSG_TEST_FAIL_PERC_30D,
        MSG_TEST_FAIL_PERC_7D,
        MSG_TEST_FAIL_PERC_1D,
        MSG_OVERBOOK_RAM,
        MSG_OVERBOOK_CORES,
        MSG_BANDWIDTH_IN,
        MSG_BANDWIDTH_OUT,
        MSG_USER_GROUP,
    ]
}


def load_local_dataset(
    *, filename: str, dataset_version: str, logger: Logger
) -> pd.DataFrame:
    """Load a dataset from a local CSV or JSON file.

    Load the dataset, validate its columns, and return it as a pandas DataFrame.

    Args:
        filename (str): Path to the local dataset file (CSV or JSON).
        dataset_version (str): The version of the dataset, used to validate expected
            columns.
        logger (Logger): Logger instance for logging information and debug messages.

    Returns:
        pd.DataFrame: The loaded and validated dataset.

    Raises:
        ConfigurationError: If the file is not found, the file extension is unsupported,
            the dataset version is not supported, or the columns are invalid.

    """
    try:
        if filename is None:
            raise ValueError("LOCAL_DATASET environment variable has not been set.")

        # Detect file extension
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        # Create dataframe based of file type
        if ext == ".csv":
            logger.info("Read data from CSV file: '%s'", filename)
            df = pd.read_csv(filename)
        elif ext == ".json":
            logger.info("Read data from JSON file: '%s'", filename)
            with open(filename) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Check columns based on dataset version
        msg_map = MSG_VALID_KEYS.get(dataset_version, None)
        if msg_map is None:
            raise ValueError(f"Dataset version {dataset_version} not supported")

        invalid_keys = set(df.columns).difference(msg_map)
        assert len(invalid_keys) == 0, f"Found invalid keys: {invalid_keys}"

        logger.debug("Uploaded dataframe:")
        logger.debug(df)
        return df

    except FileNotFoundError as e:
        raise ConfigurationError(f"File '{filename}' not found") from e
    except (ValueError, AssertionError) as e:
        raise ConfigurationError(e.args[0]) from e


def load_dataset_from_kafka_messages(
    *, consumer: KafkaConsumer, logger: Logger
) -> pd.DataFrame:
    """Read messages from a Kafka consumer and build DataFrame from them.

    Reads messages, validates their structure based on versioned schemas, and constructs
    a pandas DataFrame from the valid messages.

    Args:
        consumer (KafkaConsumer): The Kafka consumer instance to read messages from.
        logger (Logger): Logger instance for debug output.

    Returns:
        pd.DataFrame: DataFrame constructed from the validated Kafka messages.

    Raises:
        ConfigurationError: If a message has an unsupported version or contains invalid
            keys.

    """
    try:
        messages = [message.value for message in consumer]
        for message in messages:
            msg_version = message.pop(MSG_VERSION)
            msg_map = MSG_VALID_KEYS.get(msg_version, None)
            if msg_map is None:
                raise ValueError(f"Message version {msg_version} not supported")

            invalid_keys = set(message.keys()).difference(msg_map)
            assert len(invalid_keys) == 0, f"Found invalid keys: {invalid_keys}"

        df = pd.DataFrame(messages)
        logger.debug("Uploaded dataframe:")
        logger.debug(df)
        return df

    except (ValueError, AssertionError) as e:
        raise ConfigurationError(e.args[0]) from e


def load_data_from_file(*, filename: str | None, logger: Logger) -> list[dict]:
    """Load local messages from a JSON file.

    Args:
        filename (str | None): The path to the file containing the data. If None, an
            error is raised.
        logger (Logger): Logger instance for logging debug information.

    Returns:
        list[dict]: The data loaded from the file as a list of dictionaries.

    Raises:
        ConfigurationError: If the file is not found or if the filename is not provided.

    """
    try:
        if filename is None:
            raise ValueError(
                "LOCAL_IN_MESSAGES or LOCAL_OUT_MESSAGES environment variable "
                "have not been set."
            )

        with open(filename) as file:
            data = json.load(file)

        logger.debug("Loaded data: %s", data)
        return data

    except FileNotFoundError as e:
        raise ConfigurationError(f"File '{filename}' not found") from e
    except ValueError as e:
        raise ConfigurationError(e.args[0]) from e


def write_data_to_file(*, filename: str | None, data: dict[str, Any]) -> None:
    """Write a dictionary of data to a specified JSON file.

    If the file exists, the function loads its current contents (expected to be a list),
    appends the new data, and writes the updated list back to the file. If the file does
    not exist, a ConfigurationError is raised. If the filename is not provided, a
    ConfigurationError is raised.

    Args:
        filename (str | None): The path to the file where data should be written.
            Must not be None.
        data (dict[str, Any]): The dictionary data to append to the file.

    Raises:
        ConfigurationError: If the file is not found or if the filename is None.

    """
    try:
        if filename is None:
            raise ValueError(
                "LOCAL_OUT_MESSAGES environment variable has not been set."
            )
        with open(filename, "r+") as file:
            values = json.load(file)
            values.append(data)
            file.seek(0)
            file.truncate()
            json.dump(values, file, indent=4)
    except FileNotFoundError as e:
        raise ConfigurationError(f"File '{filename}' not found") from e
    except ValueError as e:
        raise ConfigurationError(e.args[0]) from e
