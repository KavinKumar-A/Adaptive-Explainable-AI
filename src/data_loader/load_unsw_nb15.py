import pandas as pd
import re


def _normalize_column_names(data):
    """
    Correction #13: Normalize dataset feature names.
    Handles variations like "Flow Duration" vs "flow_duration"
    by converting to lowercase and replacing spaces/hyphens with underscores.
    """
    normalized = {}
    for col in data.columns:
        clean = col.strip().lower()
        clean = re.sub(r'[\s\-]+', '_', clean)
        clean = re.sub(r'_+', '_', clean)
        normalized[col] = clean
    data.rename(columns=normalized, inplace=True)
    return data


def load():
    """
    Load UNSW-NB15 dataset by combining training and testing sets.
    Applies feature name normalization (Correction #13).
    """
    train_data = pd.read_csv('datasets/UNSW_NB15/UNSW_NB15_training-set.csv')
    test_data = pd.read_csv('datasets/UNSW_NB15/UNSW_NB15_testing-set.csv')
    data = pd.concat([train_data, test_data], ignore_index=True)

    # Correction #13: Normalize all column names
    data = _normalize_column_names(data)

    # AXAI-IDS Prevention: Drop data leak columns ('id', 'attack_cat')
    cols_to_drop = ['id', 'attack_cat']
    data.drop(columns=[col for col in cols_to_drop if col in data.columns], inplace=True)

    return data