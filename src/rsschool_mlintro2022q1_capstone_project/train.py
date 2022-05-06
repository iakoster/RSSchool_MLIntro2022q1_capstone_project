from pathlib import Path

import click
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .models import make_pipeline


def get_dataset(dataset_path: Path
                ) -> tuple[pd.DataFrame, pd.Series]:
    if dataset_path.suffix != '.csv':
        raise TypeError(
            'Data profiling does not support '
            'anything other than .csv data extension')
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.lower()
    df.set_index('id', inplace=True)
    return df.drop(columns='cover_type'), df['cover_type']


@click.command()
def train(
        dataset_path: Path = None,
        model: str = 'knn',
        random_state: int = 42
):
    if dataset_path is None:
        dataset_path = Path(r'data\train.csv')
    X, y = get_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    model = make_pipeline()
    model.fit(X_train, y_train)
    print(accuracy_score(y_test, model.predict(X_test)))
