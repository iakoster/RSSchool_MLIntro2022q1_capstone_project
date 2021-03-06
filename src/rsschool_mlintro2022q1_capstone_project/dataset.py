from pathlib import Path

import pandas as pd


def get_dataset(dataset_path: Path) -> pd.DataFrame:
    """Read and format Forest Cover Type dataset."""
    if dataset_path.suffix != ".csv":
        raise TypeError("Unsupported data extension: %s" % dataset_path.suffix)
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.lower()
    df.set_index("id", inplace=True)
    return df


def get_dataset_xy(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Returns features and target from dataset."""
    df = get_dataset(dataset_path)
    return df.drop(columns="cover_type"), df["cover_type"]
