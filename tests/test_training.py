import pandas as pd

from src.training.main import remove_outliers_from_dataframe


def test_remove_outliers_from_dataframe():
    # Create a sample DataFrame with outliers
    data = {
        'A': [1, -10, 3, 4, 5, 100],  # Outlier at the end
        'B': [10, 20, 30, 40, 50, 200],  # Outlier at the end
        'C': [100, 200, 300, 400, 500, 600]  # No outliers
    }
    df = pd.DataFrame(data)

    # Call the function to remove outliers
    cleaned_df = remove_outliers_from_dataframe(df)

    # Check that the outliers have been removed
    assert len(cleaned_df) == 4
    assert not any(cleaned_df['A'] == 100)
    assert not any(cleaned_df['B'] == 200)
    assert not any(cleaned_df['C'] == 600)