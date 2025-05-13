import json
import os
from logging import Logger
from typing import Any

import pandas as pd
from kafka import KafkaConsumer

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
MSG_DEP_COMPLETION_TIME = "completetion_time_s"
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
MSG_IMAGES = "images"
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
        MSG_IMAGES,
        MSG_USER_GROUP,
    ]
}


def load_local_dataset(
    *, filename: str, dataset_version: str, logger: Logger
) -> pd.DataFrame:
    """Load a dataset from a local CSV or JSON file and validate columns."""

    # Detect file extension
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    # Create dataframe based of file type
    if ext == ".csv":
        df = pd.read_csv(filename)
    elif ext == ".json":
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


def load_dataset_from_kafka_messages(
    *, consumer: KafkaConsumer, logger: Logger
) -> pd.DataFrame:
    """Read kafka messages and create a dataset from them."""
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


def load_data_from_file(*, filename: str | None, logger: Logger) -> list[dict]:
    """Load local messages from a text file."""
    with open(filename) as file:
        data = json.load(file)
    logger.debug("Loaded data: %s", data)
    return data


def write_data_to_file(
    *, filename: str | None, data: dict[str, Any], logger: Logger
) -> None:
    """Write data to file."""
    # 'a' mode appends without overwriting
    with open(filename, "r+") as file:
        values = json.load(file)
        values.append(data)
        json.dump(values, file, indent=4)
    logger.info("Message written into %s", filename)
