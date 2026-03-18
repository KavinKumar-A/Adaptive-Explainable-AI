import pandas as pd
import os
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
    Load CIC-IDS2017 dataset by combining all CSV files in the folder.
    Applies feature name normalization (Correction #13).
    """
    path = 'datasets/CIC_IDS2017/'
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = []
    for file in files:
        df = pd.read_csv(os.path.join(path, file))
        df_list.append(df)
    data = pd.concat(df_list, ignore_index=True)

    # Correction #13: Normalize all column names
    data = _normalize_column_names(data)

    return data