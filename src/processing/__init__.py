from logging import Logger

import numpy as np
import pandas as pd

from utils import (
    MSG_CPU_QUOTA,
    MSG_CPU_REQ,
    MSG_CPU_USAGE,
    MSG_DEP_COMPLETION_TIME,
    MSG_DEP_FAILED_TIME,
    MSG_DEP_TOT_FAILURES,
    MSG_DEP_UUID,
    MSG_DISK_QUOTA,
    MSG_DISK_REQ,
    MSG_DISK_USAGE,
    MSG_GPU_REQ,
    MSG_IMAGES,
    MSG_INSTANCE_QUOTA,
    MSG_INSTANCE_REQ,
    MSG_INSTANCE_USAGE,
    MSG_INSTANCES_WITH_EXACT_FLAVORS,
    MSG_OVERBOOK_CORES,
    MSG_OVERBOOK_RAM,
    MSG_PROVIDER_NAME,
    MSG_PUB_IPS_QUOTA,
    MSG_PUB_IPS_REQ,
    MSG_PUB_IPS_USAGE,
    MSG_RAM_QUOTA,
    MSG_RAM_REQ,
    MSG_RAM_USAGE,
    MSG_REGION_NAME,
    MSG_STATUS,
    MSG_STATUS_REASON,
    MSG_TEMPLATE_NAME,
    MSG_TEST_FAIL_PERC_1D,
    MSG_TEST_FAIL_PERC_7D,
    MSG_TEST_FAIL_PERC_30D,
    MSG_TIMESTAMP,
    MSG_USER_GROUP,
    MSG_VOL_QUOTA,
    MSG_VOL_REQ,
    MSG_VOL_USAGE,
)

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
DF_MAX_DEP_TIME = "max_success_time"
DF_MIN_DEP_TIME = "min_success_time"

STATUS_CREATE_COMPLETED = "CREATE_COMPLETE"
STATUS_CREATE_FAILED = "CREATE_FAILED"
STATUS_CREATE_COMPLETED_VALUE = 0
STATUS_CREATE_FAILED_VALUE = 1
STATUS_MAP = {
    STATUS_CREATE_COMPLETED: STATUS_CREATE_COMPLETED_VALUE,
    STATUS_CREATE_FAILED: STATUS_CREATE_FAILED_VALUE,
}


def calculate_derived_properties(
    *, df: pd.DataFrame, complex_templates: list[str]
) -> pd.DataFrame:
    """From message inputs, calculate derived properties.

    Concatenate provider and region name.
    CPU diff: difference between Maximum, used and requested.
    RAM diff: difference between Maximum, used and requested.
    Disk diff: difference between Maximum, used and requested.
    Instances diff: difference between Maximum, used and requested.
    Volumes diff: difference between Maximum, used and requested.
    Public IPs diff: difference between Maximum, used and requested.
    Template complexity depends on the chosen template.
    """
    df[DF_PROVIDER] = df[[MSG_PROVIDER_NAME, MSG_REGION_NAME]].agg("-".join, axis=1)
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
    df[DF_GPU] = df[MSG_GPU_REQ]
    df[DF_COMPLEX] = df[MSG_TEMPLATE_NAME].isin(complex_templates).astype(int)
    return df


def preprocessing(
    *,
    df: pd.DataFrame,
    complex_templates: list[str],
    logger: Logger,
) -> pd.DataFrame:
    """Pre-process data in the dataframe.

    Calculate CPU diff as the difference between Quota (maximum value), used and
    requested. The same goes for RAM, disk, instances, volumes and public IPs.

    Mapped the success and failed creation events to integers.

    Based on the template, set a complexity value.

    The deployment time, when the deployment fails is the average failure time of every
    attempt.

    TODO: ...

    Return dataframe with only desired features.
    """
    logger.info("Pre-process data")
    logger.debug("Initial Dataframe:\n%s", df)
    if df.empty:
        logger.warning("Received an empty dataframe")
        return df

    # Map STATUS to integer and Convert df[DF_TIMESTAMP]. This may generate NaN values.
    # Remove them.
    df.drop([MSG_DEP_UUID, MSG_STATUS_REASON, MSG_USER_GROUP], axis=1, inplace=True)
    df[DF_STATUS] = df[MSG_STATUS].map(STATUS_MAP)
    df[DF_TIMESTAMP] = pd.to_datetime(df[MSG_TIMESTAMP], errors="coerce")
    # df[DF_DEP_TIME] is the completion time if the deployment successful otherwise it
    # is the average failure time
    df[DF_DEP_TIME] = np.where(
        df[MSG_DEP_COMPLETION_TIME] != 0.0,
        df[MSG_DEP_COMPLETION_TIME],
        np.where(
            df[MSG_DEP_TOT_FAILURES] != 0,
            df[MSG_DEP_FAILED_TIME] / df[MSG_DEP_TOT_FAILURES],
            np.nan,
        ),
    )
    df.dropna(inplace=True)
    if df.empty:
        logger.warning("Dropping NaN and None generated an empty dataframe")
        return df

    df = calculate_derived_properties(df=df, complex_templates=complex_templates)

    # Calculate historical features.
    grouped = df.groupby([DF_PROVIDER, MSG_TEMPLATE_NAME])
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
    df[DF_MIN_DEP_TIME] = df.apply(
        lambda row: calculate_min_success_time(
            grouped.get_group((row[DF_PROVIDER], row[MSG_TEMPLATE_NAME])), row
        ),
        axis=1,
    )
    df[DF_MAX_DEP_TIME] = df.apply(
        lambda row: calculate_max_success_time(
            grouped.get_group((row[DF_PROVIDER], row[MSG_TEMPLATE_NAME])), row
        ),
        axis=1,
    )

    logger.debug("Final dataframe: %s", df)
    logger.info("Pre-process completed")
    return df


def calculate_failure_percentage(group: pd.DataFrame, row: pd.Series) -> float:
    """Function to calculate the failure percentage"""
    mask = (group[DF_TIMESTAMP] <= row[DF_TIMESTAMP]) & (
        group[DF_TIMESTAMP] > row[DF_TIMESTAMP] - pd.Timedelta(days=30)
    )
    filtered_group = group[mask]
    return filtered_group[DF_STATUS].mean()


def calculate_avg_success_time(group: pd.DataFrame, row: pd.Series) -> float:
    """Function to calculate the average success time"""
    mask = (
        (group[DF_TIMESTAMP] <= row[DF_TIMESTAMP])
        & (group[DF_TIMESTAMP] > row[DF_TIMESTAMP] - pd.Timedelta(days=30))
        & (group[DF_STATUS] == 0)
    )
    filtered_group = group[mask]
    return filtered_group[DF_DEP_TIME].mean() if not filtered_group.empty else 0.0


def calculate_avg_failure_time(group: pd.DataFrame, row: pd.Series) -> float:
    """Function to calculate average failure time"""
    mask = (
        (group[DF_TIMESTAMP] <= row[DF_TIMESTAMP])
        & (group[DF_TIMESTAMP] > row[DF_TIMESTAMP] - pd.Timedelta(days=30))
        & (group[DF_STATUS] == 1)
    )
    filtered_group = group[mask]
    return filtered_group[DF_DEP_TIME].mean() if not filtered_group.empty else 0.0


def calculate_max_success_time(group: pd.DataFrame, row: pd.Series) -> float:
    """Function to calculate maximum success time"""
    mask = (
        (group[DF_TIMESTAMP] <= row[DF_TIMESTAMP])
        & (group[DF_TIMESTAMP] > row[DF_TIMESTAMP] - pd.Timedelta(days=30))
        & (group[DF_STATUS] == 0)
    )
    filtered_group = group[mask]
    return filtered_group[DF_DEP_TIME].max() if not filtered_group.empty else 0.0


def calculate_min_success_time(group: pd.DataFrame, row: pd.Series) -> float:
    """Function to calculate minimum success time.

    A valid minimum success time must be greater than 0.
    """
    mask = (
        (group[DF_TIMESTAMP] <= row[DF_TIMESTAMP])
        & (group[DF_TIMESTAMP] > row[DF_TIMESTAMP] - pd.Timedelta(days=30))
        & (group[DF_STATUS] == 0)
        & (group[MSG_DEP_COMPLETION_TIME] > 0)
    )
    filtered_group = group[mask]
    return filtered_group[DF_DEP_TIME].min() if not filtered_group.empty else 0.0
