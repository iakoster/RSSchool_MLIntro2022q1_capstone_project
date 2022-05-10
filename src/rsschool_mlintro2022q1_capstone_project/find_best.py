import warnings
from pathlib import Path
from joblib import dump
from typing import Union, Mapping

import click
import numpy as np
import numpy.typing as npt
import pandas as pd

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV, cross_validate

from .dataset import get_dataset_xy
from .models import create_pipeline, get_metrics
from .settings import (
    DATASET_PATH_TRAIN,
    STD_BEST_MODEL_PATH,
)


warnings.simplefilter("ignore", append=True)


def eval_metrics(
    est: Pipeline, x: pd.DataFrame, y: Union[pd.Series, npt.NDArray[np.int_]]
) -> dict[str, float]:
    """Make a prediction and calculate metrics."""
    accuracy, f1_score, roc_auc_ovr = get_metrics(
        y, est.predict(x), est.predict_proba(x)
    )
    return {
        "accuracy": accuracy,
        "f1_score": f1_score,
        "roc_auc_ovr": roc_auc_ovr,
    }


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
    default=STD_BEST_MODEL_PATH,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "-m",
    "--model",
    default="knn",
    type=click.Choice(["knn", "forest"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--scale",
    is_flag=True,
    show_default=True,
)
@click.option(
    "--scaler",
    default="standard",
    type=click.Choice(["standard", "minmax"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--normalize",
    is_flag=True,
    show_default=True,
)
@click.option(
    "--n-jobs",
    default=1,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
def find_best(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    model: str,
    scale: bool,
    scaler: str,
    normalize: bool,
    n_jobs: int,
) -> None:
    """Search the best hyperparameters for the model."""
    features, target = get_dataset_xy(dataset_path)
    space = get_space(model)

    try:
        pipeline = create_pipeline(
            model=model,
            random_state=random_state,
            scale=scale,
            scaler=scaler,
            normalize=normalize,
            k_best=54,
            n_jobs=n_jobs,
        )
    except Exception as exc:
        raise click.BadParameter(
            f"Raised exception while setting pipeline: {exc}"
        )

    with mlflow.start_run():
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_outer = KFold(n_splits=5, shuffle=True, random_state=random_state)

        search = GridSearchCV(
            pipeline, space, scoring="accuracy", cv=cv_inner, n_jobs=n_jobs
        )
        nested_result = cross_validate(
            search,
            features,
            target,
            scoring=eval_metrics,
            cv=cv_outer,
            n_jobs=n_jobs,
        )
        accuracy, f1_score, roc_auc_ovr = (
            nested_result["test_accuracy"].mean(),
            nested_result["test_f1_score"].mean(),
            nested_result["test_roc_auc_ovr"].mean(),
        )

        search.fit(features, target)
        best_params = search.best_params_
        click.echo(f"Best parameters: {search.best_params_}")
        k_best = best_params.pop("k_best__k")
        model_kw_fmt = {k.split("__")[1]: v for k, v in best_params.items()}

        pipeline = create_pipeline(
            model=model,
            random_state=random_state,
            scale=scale,
            scaler=scaler,
            normalize=normalize,
            k_best=k_best,
            model_kw=model_kw_fmt,
            n_jobs=n_jobs,
        )
        pipeline.fit(features, target)
        dump(pipeline, save_model_path)

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "f1_score": f1_score,
                "roc_auc_ovr": roc_auc_ovr,
            }
        )
        mlflow.sklearn.log_model(pipeline, model)
        mlflow.log_params(
            {
                "model": model,
                "random_state": random_state,
                "k_folds": 5,
                "use_scaler": scale,
                "scaler": scaler,
                "normalize": normalize,
                "k_best": k_best,
            }
        )
        mlflow.log_param(
            "model_params",
            ", ".join(f"{k}={v}" for k, v in model_kw_fmt.items()),
        )

        click.echo(
            f"Metrics. "
            f"Accuracy: {accuracy:.6f}, "
            f"F1 score: {f1_score:.6f}, "
            f"ROC AUC OVR: {roc_auc_ovr:.6f}"
        )
        click.echo(f"Model saved in {save_model_path}")


def get_space(
    model: str,
) -> Mapping[str, Union[list[int], list[str]]]:
    """Get space of the hyperparameters for search."""
    assert model in ("knn", "forest"), "invalid model %s" % model

    space: dict[str, Union[list[int], list[str]]]
    space = {"k_best__k": [5, 10, 15, 20, 25, 30, 40, 50, 54]}
    if model == "knn":
        space.update(
            {
                f"{model}__n_neighbors": [1, 5, 10, 15, 20],
                f"{model}__metric": ["euclidean", "manhattan", "minkowski"],
            }
        )
    elif model == "forest":
        space.update(
            {
                f"{model}__n_estimators": [10, 20, 40, 70, 100],
                f"{model}__criterion": ["gini", "entropy"],
                f"{model}__max_depth": [5, 10, 20, 35, 50, 75, 100],
            }
        )

    return space
