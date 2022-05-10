from pathlib import Path
from joblib import load

import click
import pandas as pd

from .dataset import get_dataset
from .settings import (
    STD_BEST_MODEL_PATH,
    DATASET_PATH_TEST,
    PREDICTION_PATH,
)


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default=DATASET_PATH_TEST,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-m",
    "--model-path",
    default=STD_BEST_MODEL_PATH,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-prediction-path",
    default=PREDICTION_PATH,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def predict(dataset_path: Path, model_path: Path, save_prediction_path: Path) -> None:
    features = get_dataset(dataset_path)
    prediction = pd.DataFrame(
        data=load(model_path).predict(features),
        index=features.index,
        columns=["Cover_Type"],
    )
    prediction.index.name = "Id"
    prediction.to_csv(save_prediction_path)
