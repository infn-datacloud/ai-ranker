import pandas as pd
from src.training.main import remove_outliers_from_dataframe, remove_outliers


def test_remove_outliers_from_dataframe():
    # Create a DataFrame with clear outliers
    data = {
        'A': [1, -10, 3, 4, 5, 100],  # Outlier at the end
        'B': [10, 20, 30, 40, 50, 200],  # Outlier at the end
        'C': [100, 200, 300, 400, 500, 600]  # No outliers in this column
    }
    df = pd.DataFrame(data)
    cleaned_df = remove_outliers_from_dataframe(df)

    # Outliers should be removed
    assert len(cleaned_df) == 4
    assert 100 not in cleaned_df['A'].values
    assert 200 not in cleaned_df['B'].values
    assert 600 not in cleaned_df['B'].values


def test_remove_outliers_from_dataframe_no_outliers():
    # All values within a reasonable range, no outliers expected
    df = pd.DataFrame({
        'A': [10, 12, 14, 16, 18, 20],
        'B': [5, 7, 9, 11, 13, 15]
    })
    cleaned_df = remove_outliers_from_dataframe(df)
    pd.testing.assert_frame_equal(cleaned_df, df)


def test_remove_outliers_from_dataframe_constant_columns():
    # Columns with constant values should not generate outliers
    df = pd.DataFrame({
        'A': [5, 5, 5, 5, 5],
        'B': [10, 10, 10, 10, 10]
    })
    cleaned_df = remove_outliers_from_dataframe(df)
    pd.testing.assert_frame_equal(cleaned_df, df)


def test_remove_outliers_from_dataframe_single_extreme_value():
    # One extreme value in an otherwise uniform column
    df = pd.DataFrame({
        'A': [1, 1, 1, 1, 1000],
        'B': [2, 2, 2, 2, 2]
    })
    cleaned_df = remove_outliers_from_dataframe(df)
    assert len(cleaned_df) == 4
    assert 1000 not in cleaned_df['A'].values


def test_remove_outliers_from_dataframe_empty():
    # Edge case: empty DataFrame
    df = pd.DataFrame(columns=['A', 'B'])
    cleaned_df = remove_outliers_from_dataframe(df)
    pd.testing.assert_frame_equal(cleaned_df, df)


def test_remove_outliers_from_dataframe_with_nan():
    # NaN rows should be handled correctly (we drop them before passing to the function)
    df = pd.DataFrame({
        'A': [1, 2, 3, None, 100],
        'B': [10, 20, 30, 40, 200]
    })
    df_clean = df.dropna()
    cleaned_df = remove_outliers_from_dataframe(df_clean)
    assert 100 not in cleaned_df['A'].values
    assert 200 not in cleaned_df['B'].values


def test_remove_outliers_from_dataframe_custom_params():
    # Custom IQR parameters to test different sensitivity to outliers
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 100]
    })

    df_clean_strict = remove_outliers_from_dataframe(df, q1=0.25, q3=0.75, k=1.0)
    assert 100 not in df_clean_strict['A'].values
    assert len(df_clean_strict) <= 5

    df_clean_very_strict = remove_outliers_from_dataframe(df, q1=0.4, q3=0.6, k=1.0)
    assert len(df_clean_very_strict) < len(df)

    df_clean_no_removal = remove_outliers_from_dataframe(df, q1=0.25, q3=0.75, k=100.0)
    assert len(df_clean_no_removal) == 6
    assert 100 in df_clean_no_removal['A'].values


def test_remove_outliers_from_dataframe_single_column():
    # Handle single-column DataFrame
    df = pd.DataFrame({'A': [1, 2, 3, 4, 1000]})
    cleaned_df = remove_outliers_from_dataframe(df)
    assert 1000 not in cleaned_df['A'].values


def test_remove_outliers_from_dataframe_negative_positive():
    # Mix of negative and positive values with outliers on both ends
    df = pd.DataFrame({
        'A': [-100, -10, 0, 10, 20, 200],
        'B': [5, 5, 5, 5, 5, 5]
    })
    cleaned_df = remove_outliers_from_dataframe(df)
    assert -100 not in cleaned_df['A'].values
    assert 200 not in cleaned_df['A'].values
    assert all(cleaned_df['B'] == 5)


def test_remove_outliers_combined_basic():
    # Outliers in both X and Y should be removed together
    x = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 200]
    })
    y = pd.DataFrame({
        'target': [0, 1, 0, 1, 0, 1]
    })
    x_clean, y_clean = remove_outliers(x, y)
    assert len(x_clean) == 5
    assert 100 not in x_clean['A'].values
    assert 200 not in x_clean['B'].values
    assert len(y_clean) == len(x_clean)


def test_remove_outliers_combined_no_outliers():
    # No outliers in either X or Y
    x = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    y = pd.DataFrame({
        'target': [1, 0, 1, 0, 1]
    })
    x_clean, y_clean = remove_outliers(x, y)
    pd.testing.assert_frame_equal(x, x_clean)
    pd.testing.assert_frame_equal(y, y_clean)


def test_remove_outliers_combined_outlier_in_y():
    # Outlier only in Y column
    x = pd.DataFrame({
        'A': [1, 2, 3, 4, 5]
    })
    y = pd.DataFrame({
        'target': [10, 20, 30, 40, 9999]
    })
    x_clean, y_clean = remove_outliers(x, y)
    assert 9999 not in y_clean['target'].values
    assert len(x_clean) == len(y_clean) == 4



def test_remove_outliers_combined_constant_columns():
    # Constant values in both X and Y, nothing to remove
    x = pd.DataFrame({
        'A': [5, 5, 5, 5]
    })
    y = pd.DataFrame({
        'target': [1, 1, 1, 1]
    })
    x_clean, y_clean = remove_outliers(x, y)
    pd.testing.assert_frame_equal(x, x_clean)
    pd.testing.assert_frame_equal(y, y_clean)


def test_remove_outliers_combined_column_names_preserved():
    # Ensure that column names are preserved after filtering
    x = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 100]
    })
    y = pd.DataFrame({
        'label': [0, 1, 0, 1, 1]
    })
    x_clean, y_clean = remove_outliers(x, y)
    assert list(x_clean.columns) == ['feature1']
    assert list(y_clean.columns) == ['label']


