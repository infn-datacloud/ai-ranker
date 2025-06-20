from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.processing import (
    DF_AVG_FAIL_TIME,
    DF_AVG_SUCCESS_TIME,
    DF_COMPLEX,
    DF_CPU_DIFF,
    DF_DEP_TIME,
    DF_DISK_DIFF,
    DF_FAIL_PERC,
    DF_GPU,
    DF_INSTANCE_DIFF,
    DF_MAX_DEP_TIME,
    DF_MIN_DEP_TIME,
    DF_PUB_IPS_DIFF,
    DF_RAM_DIFF,
    DF_STATUS,
    DF_TIMESTAMP,
    DF_VOL_DIFF,
    STATUS_CREATE_COMPLETED,
    calculate_avg_failure_time,
    calculate_avg_success_time,
    calculate_derived_properties,
    calculate_failure_percentage,
    calculate_max_success_time,
    calculate_min_success_time,
    preprocessing,
)
from src.utils import (
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
    MSG_STATUS,
    MSG_STATUS_REASON,
    MSG_TEMPLATE_NAME,
    MSG_TIMESTAMP,
    MSG_USER_GROUP,
    MSG_VOL_QUOTA,
    MSG_VOL_REQ,
    MSG_VOL_USAGE,
)

BASE_DATA = {
    MSG_PROVIDER_NAME: ["provider1"],
    MSG_REGION_NAME: ["region1"],
    MSG_CPU_QUOTA: [10],
    MSG_CPU_USAGE: [5],
    MSG_CPU_REQ: [3],
    MSG_RAM_QUOTA: [32],
    MSG_RAM_USAGE: [16],
    MSG_RAM_REQ: [8],
    MSG_DISK_QUOTA: [100],
    MSG_DISK_USAGE: [50],
    MSG_DISK_REQ: [20],
    MSG_INSTANCE_QUOTA: [10],
    MSG_INSTANCE_USAGE: [5],
    MSG_INSTANCE_REQ: [2],
    MSG_VOL_QUOTA: [50],
    MSG_VOL_USAGE: [25],
    MSG_VOL_REQ: [10],
    MSG_PUB_IPS_QUOTA: [5],
    MSG_PUB_IPS_USAGE: [2],
    MSG_PUB_IPS_REQ: [1],
    MSG_GPU_REQ: [1],
    MSG_TEMPLATE_NAME: ["tpl1"],
    MSG_STATUS: [STATUS_CREATE_COMPLETED],
    MSG_TIMESTAMP: [pd.Timestamp("2024-01-01")],
    MSG_DEP_COMPLETION_TIME: [30.0],
    MSG_DEP_FAILED_TIME: [0.0],
    MSG_DEP_TOT_FAILURES: [0],
    MSG_DEP_UUID: ["uuid"],
    MSG_STATUS_REASON: ["ok"],
    MSG_USER_GROUP: ["group"],
}


@pytest.fixture
def base_df():
    return pd.DataFrame(BASE_DATA.copy())


def test_calculate_derived_properties(base_df):
    df = calculate_derived_properties(df=base_df.copy(), complex_templates=["tpl1"])
    assert DF_CPU_DIFF in df.columns
    assert df[DF_CPU_DIFF].iloc[0] == 2  # (10 - 5) - 3
    assert df[DF_COMPLEX].iloc[0] == 1


def test_preprocessing_returns_correct_columns(base_df):
    logger = MagicMock()
    df = preprocessing(df=base_df.copy(), complex_templates=["tpl1"], logger=logger)
    expected_cols = {
        DF_CPU_DIFF,
        DF_RAM_DIFF,
        DF_DISK_DIFF,
        DF_INSTANCE_DIFF,
        DF_VOL_DIFF,
        DF_PUB_IPS_DIFF,
        DF_GPU,
        DF_COMPLEX,
        DF_FAIL_PERC,
        DF_AVG_SUCCESS_TIME,
        DF_AVG_FAIL_TIME,
        DF_MAX_DEP_TIME,
        DF_MIN_DEP_TIME,
    }
    assert expected_cols.issubset(df.columns)
    logger.info.assert_called()
    logger.debug.assert_called()


def test_preprocessing_handles_empty_df():
    logger = MagicMock()
    df = pd.DataFrame()
    out = preprocessing(df=df, complex_templates=["tpl1"], logger=logger)
    assert out.empty
    logger.warning.assert_called_with("Received an empty dataframe")


def test_preprocessing_drops_nans_and_returns_empty():
    logger = MagicMock()
    df = pd.DataFrame({k: [np.nan] for k in BASE_DATA.keys()})
    out = preprocessing(df=df, complex_templates=["tpl1"], logger=logger)
    assert out.empty
    logger.warning.assert_called_with(
        "Dropping NaN and None generated an empty dataframe"
    )


def test_calculate_failure_percentage():
    df = pd.DataFrame(
        {DF_TIMESTAMP: pd.date_range("2024-01-01", periods=3), DF_STATUS: [0, 1, 1]}
    )
    row = df.iloc[2]
    perc = calculate_failure_percentage(df, row)
    assert 0 <= perc <= 1


def test_calculate_avg_success_time():
    df = pd.DataFrame(
        {
            DF_TIMESTAMP: pd.date_range("2024-01-01", periods=3),
            DF_STATUS: [0, 0, 0],
            DF_DEP_TIME: [10.0, 20.0, 30.0],
        }
    )
    row = df.iloc[2]
    assert calculate_avg_success_time(df, row) == 20.0


def test_calculate_avg_failure_time():
    df = pd.DataFrame(
        {
            DF_TIMESTAMP: pd.date_range("2024-01-01", periods=3),
            DF_STATUS: [1, 1, 1],
            DF_DEP_TIME: [5.0, 10.0, 15.0],
        }
    )
    row = df.iloc[2]
    assert calculate_avg_failure_time(df, row) == 10.0


def test_calculate_max_success_time():
    df = pd.DataFrame(
        {
            DF_TIMESTAMP: pd.date_range("2024-01-01", periods=3),
            DF_STATUS: [0, 0, 0],
            DF_DEP_TIME: [10.0, 20.0, 30.0],
        }
    )
    row = df.iloc[2]
    assert calculate_max_success_time(df, row) == 30.0


def test_calculate_min_success_time():
    df = pd.DataFrame(
        {
            DF_TIMESTAMP: pd.date_range("2024-01-01", periods=3),
            DF_STATUS: [0, 0, 0],
            DF_DEP_TIME: [5.0, 10.0, 15.0],
            MSG_DEP_COMPLETION_TIME: [5.0, 10.0, 15.0],
        }
    )
    row = df.iloc[2]
    assert calculate_min_success_time(df, row) == 5.0
