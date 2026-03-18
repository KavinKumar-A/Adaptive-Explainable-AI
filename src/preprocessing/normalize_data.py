import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize(data, target_col=None):
    """
    Normalize numeric features using MinMaxScaler to [0, 1] range.
    Per AXAI-IDS methodology (Paper §7.1):
    'Min-Max Scaler is applied across all input feature columns,
     compressing the disparate values into a standardized [0, 1] range.'
    Excludes target column from normalization.
    """
    numeric_cols = data.select_dtypes(include=['number']).columns

    # Exclude target column from normalization
    if target_col and target_col in numeric_cols:
        numeric_cols = numeric_cols.drop(target_col)

    # Apply MinMaxScaler to feature columns only
    if len(numeric_cols) > 0:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        return data, scaler
    else:
        return data, None