from pathlib import Path
from joblib import dump

import click
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

from .models import create_pipeline, get_metrics
from .settings import DATASET_PATH_TRAIN, STD_MODEL_PATH


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


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default=DATASET_PATH_TRAIN,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default=STD_MODEL_PATH,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "-m",
    "--model",
    default="knn",
    type=click.Choice(['knn', 'forest'], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--k-folds",
    default=5,
    type=click.IntRange(1, min_open=True),
    show_default=True,
)
@click.option(
    "--shuffle-folds/--no-shuffle-folds",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--knn-neighbors",
    default=5,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
@click.option(
    "--forest-n-estimators",
    default=100,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
@click.option(
    "--forest-criterion",
    default='gini',
    type=click.Choice(
        ['gini', 'entropy'], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--forest-max-depth",
    default=50,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
def train(
        dataset_path: Path,
        save_model_path: Path,
        model: str,
        random_state: int,
        k_folds: int,
        shuffle_folds: bool,
        knn_neighbors: int,
        forest_n_estimators: int,
        forest_criterion: str,
        forest_max_depth: int
):
    X, y = get_dataset_xy(dataset_path)
    pipeline = create_pipeline(
        model=model,
        random_state=random_state,
        knn_neighbors=knn_neighbors,
        forest_n_estimators=forest_n_estimators,
        forest_criterion=forest_criterion,
        forest_max_depth=forest_max_depth
    )
    kf = KFold(
        n_splits=k_folds,
        shuffle=shuffle_folds,
        random_state=random_state if shuffle_folds else None
    )

    accuracy_folds, f1_folds, precision_folds = [], [], []
    accuracy_best, f1_best, precision_best = 0, 0, 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        pipeline.fit(X_train, y_train)
        accuracy, f1, precision = get_metrics(
            y_test, pipeline.predict(X_test))
        accuracy_folds.append(accuracy)
        f1_folds.append(f1)
        precision_folds.append(precision)
        if accuracy > accuracy_best:
            accuracy_best, f1_best, precision_best = \
                accuracy, f1, precision
            dump(pipeline, save_model_path)
    click.echo(
        f'Mean metrics. '
        f'Accuracy: {np.mean(accuracy_folds):.6f}, '
        f'F1 score: {np.mean(f1_folds):.6f}, '
        f'Precision: {np.mean(precision_folds):.6f}'
    )
    click.echo(
        f'Best metrics. '
        f'Accuracy: {accuracy_best:.6f}, '
        f'F1 score: {f1_best:.6f}, '
        f'Precision: {precision_best:.6f}'
    )
    click.echo(f'Best model saved in {save_model_path}')

