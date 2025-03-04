import json
from logging import Logger

import numpy as np
import pandas as pd
from kafka import KafkaConsumer, TopicPartition

from settings import InferenceSettings, TrainingSettings

DF_CPU_DIFF = "cpu_diff"
DF_RAM_DIFF = "ram_diff"
DF_DISK_DIFF = "storage_diff"
DF_INSTANCE_DIFF = "instances_diff"
DF_VOL_DIFF = "volumes_diff"
DF_PUB_IPS_DIFF = "floatingips_diff"
DF_GPU = "gpu"
DF_STATUS = "status"
DF_COMPLEX = "complexity"
DF_DEP_TIME = "deployment_time"
DF_PROVIDER = "provider"
DF_TIMESTAMP = "timestamp"
DF_FAIL_PERC = "failure_percentage"
DF_AVG_SUCCESS_TIME = "avg_success_time"
DF_AVG_FAIL_TIME = "avg_failure_time"

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
MSG_TEMPLATE_NAME = "template_name"
MSG_DEP_COMPLETION_TIME = "completed_time"
MSG_DEP_FAILED_TIME = "tot_failed_time"
MSG_DEP_TOT_FAILURES = "n_failures"
MSG_PROVIDER_NAME = "provider_name"
MSG_REGION_NAME = "region_name"
MSG_TIMESTAMP = "timestamp"

STATUS_CREATE_COMPLETED = "CREATE_COMPLETED"
STATUS_CREATE_FAILED = "CREATE_FAILED"
STATUS_CREATE_COMPLETED_VALUE = 0
STATUS_CREATE_FAILED_VALUE = 1
STATUS_MAP = {
    STATUS_CREATE_COMPLETED: STATUS_CREATE_COMPLETED_VALUE,
    STATUS_CREATE_FAILED: STATUS_CREATE_FAILED_VALUE,
}


def load_local_dataset(*, filename: str, logger: Logger) -> pd.DataFrame:
    """Upload from local file the dataset."""
    try:
        df = pd.read_csv(f"{filename}")
    except FileNotFoundError:
        logger.error("File %s not found", filename)
        exit(1)
    return df


def load_dataset_from_kafka(
    kafka_server_url: str, topic: str, partition: int, offset: int = 0
):
    """Load from kafka the dataset."""
    consumer = KafkaConsumer(
        # topic,
        bootstrap_servers=kafka_server_url,
        auto_offset_reset="earliest",
        # enable_auto_commit=False,
        value_deserializer=lambda x: json.loads(
            x.decode("utf-8")
        ),  # deserializza il JSON
        consumer_timeout_ms=500,
    )

    tp = TopicPartition(topic, partition)
    consumer.assign([tp])
    consumer.seek(tp, offset)
    l_data = [message.value for message in consumer]
    df = pd.DataFrame(l_data)
    return df


def load_dataset(
    *, settings: TrainingSettings | InferenceSettings, logger: Logger
) -> pd.DataFrame:
    """Load the dataset from a local one or from kafka."""
    if settings.LOCAL_MODE:
        if settings.LOCAL_DATASET is None:
            logger.error("LOCAL_DATASET environment variable has not been set.")
            exit(1)
        return load_local_dataset(filename=settings.LOCAL_DATASET, logger=logger)
    return load_dataset_from_kafka(
        kafka_server_url=str(settings.KAFKA_URL),
        topic=settings.KAFKA_TRAINING_TOPIC,
        partition=0,
        offset=765,
    )


def preprocessing(
    *,
    df: pd.DataFrame,
    complex_templates: list[str],
    final_features: list[str],
    logger: Logger,
) -> pd.DataFrame:
    """Pre-process data in the dataframe.

    Calculate CPU diff as the difference between Quota (maximum value), used and
    requested. The same goes for RAM, disk, instances, volumes and public IPs.

    Map creation success and failed creation evente to integers.

    Based on the template, set a complexity value.

    The deployment time, when the deployment fails is the average failure time of every
    attempt.

    TODO: ...

    Return dataframe with only desired features.
    """
    logger.info("Pre-process data")
    logger.debug("Initial Dataframe:\n%s", df)

    df[DF_CPU_DIFF] = (df[MSG_CPU_QUOTA] - df[MSG_CPU_USAGE]) - df[MSG_CPU_REQ]
    df[DF_RAM_DIFF] = (df[MSG_RAM_QUOTA] - df[MSG_RAM_USAGE]) - df[MSG_RAM_REQ]
    df[DF_DISK_DIFF] = (df[MSG_DISK_QUOTA] - df[MSG_DISK_USAGE]) - df[MSG_DISK_REQ]
    df[DF_INSTANCE_DIFF] = (df[MSG_INSTANCE_QUOTA] - df[MSG_INSTANCE_USAGE]) - df[
        MSG_INSTANCE_REQ
    ]
    df[DF_VOL_DIFF] = (df[MSG_VOL_QUOTA] - df[MSG_VOL_USAGE]) - df[MSG_VOL_REQ]
    df[DF_PUB_IPS_DIFF] = (df[MSG_PUB_IPS_QUOTA] - df[MSG_PUB_IPS_USAGE]) - df[
        MSG_PUB_IPS_REQ
    ]
    df[DF_GPU] = df[MSG_GPU_REQ].astype(bool).astype(float)
    df[DF_STATUS] = df[MSG_STATUS].map(STATUS_MAP).astype(int)
    df[DF_COMPLEX] = df[MSG_TEMPLATE_NAME].isin(complex_templates).astype(float)
    df[DF_DEP_TIME] = np.where(
        df[MSG_DEP_COMPLETION_TIME] != 0.0,
        df[MSG_DEP_COMPLETION_TIME],
        df[MSG_DEP_FAILED_TIME] / df[MSG_DEP_TOT_FAILURES],
    )
    df[DF_PROVIDER] = f"{df[MSG_PROVIDER_NAME]}-{df[MSG_REGION_NAME]}"
    df[DF_TIMESTAMP] = pd.to_datetime(df[MSG_TIMESTAMP])

    grouped = df.groupby([DF_PROVIDER, MSG_TEMPLATE_NAME])
    df = df.copy()  # <-- perchè qua devi fare la copia?
    df[DF_FAIL_PERC] = df.apply(
        lambda row: calculate_failure_percentage(
            grouped.get_group((row[DF_PROVIDER], row[MSG_TEMPLATE_NAME])), row
        ),
        axis=1,
    )
    df[DF_AVG_SUCCESS_TIME] = df.apply(
        lambda row: calculate_avg_success_time(
            grouped.get_group((row[DF_PROVIDER], row[MSG_TEMPLATE_NAME])), row
        ),
        axis=1,
    )
    df[DF_AVG_FAIL_TIME] = df.apply(
        lambda row: calculate_avg_failure_time(
            grouped.get_group((row[DF_PROVIDER], row[MSG_TEMPLATE_NAME])), row
        ),
        axis=1,
    )

    df = df[final_features]

    logger.info("Pre-process completed")
    logger.debug("Return only columns: %s", final_features)
    logger.debug("Final dataframe: %s", df)

    return df


def calculate_failure_percentage(group, row):
    mask = (group[DF_TIMESTAMP] <= row[DF_TIMESTAMP]) & (
        group[DF_TIMESTAMP] > row[DF_TIMESTAMP] - pd.Timedelta(days=90)
    )
    filtered_group = group[mask]
    return filtered_group[DF_STATUS].mean() if not filtered_group.empty else None


def calculate_avg_success_time(group, row):
    mask = (
        (group[DF_TIMESTAMP] <= row[DF_TIMESTAMP])
        & (group[DF_TIMESTAMP] > row[DF_TIMESTAMP] - pd.Timedelta(days=90))
        & (group[DF_STATUS] == 0)
    )
    filtered_group = group[mask]
    return filtered_group[DF_DEP_TIME].mean() if not filtered_group.empty else 0.0


def calculate_avg_failure_time(group, row):
    mask = (
        (group[DF_TIMESTAMP] <= row[DF_TIMESTAMP])
        & (group[DF_TIMESTAMP] > row[DF_TIMESTAMP] - pd.Timedelta(days=90))
        & (group[DF_STATUS] == 1)
    )
    filtered_group = group[mask]
    return filtered_group[DF_DEP_TIME].mean() if not filtered_group.empty else 0.0
