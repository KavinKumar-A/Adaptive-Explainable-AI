import numpy as np


def clean(data):
    """
    Clean the dataset per AXAI-IDS methodology (Paper §7.1):
    1. Replace infinity values with NaN.
    2. Impute NaN values using the median of each feature column.
    3. Remove duplicate rows.
    """
    # Step 1: Replace infinite values with NaN
    data = data.replace([np.inf, -np.inf], np.nan)

    # Step 2: Median imputation for numeric columns (Paper §7.1)
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].median())

    # Step 3: Remove exact duplicate rows
    data = data.drop_duplicates()

    return data