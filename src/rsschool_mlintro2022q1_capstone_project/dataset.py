from pathlib import Path

import pandas as pd


def get_dataset(
        dataset_path: Path
) -> pd.DataFrame:
    if dataset_path.suffix != '.csv':
        raise TypeError(
            'Data profiling does not support '
            'anything other than .csv data extension')
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.lower()
    df.set_index('id', inplace=True)
    return df


def get_dataset_xy(
        dataset_path: Path
) -> tuple[pd.DataFrame, pd.Series]:
    df = get_dataset(dataset_path)
    return df.drop(columns='cover_type'), df['cover_type']