import configparser
import warnings
from pathlib import Path
from joblib import dump
from typing import Any

import click
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold

from .models import create_pipeline, get_metrics
from .settings import (
    DATASET_PATH_TRAIN,
    STD_MODEL_PATH,
    STD_CFG_PATH
)


warnings.simplefilter("ignore")


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


def format_kwargs(*params: tuple[str, str, str]
                  ) -> dict[str, Any]:
    params_kw = {}
    for name, type_, value in params:
        params_kw[name.replace('-', '_')] = eval(type_)(value)
    return params_kw


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
    "--parallel",
    is_flag=True,
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
    "--model-kw",
    nargs=3,
    type=click.Tuple([str, str, str]),
    multiple=True,
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
    type=click.Choice(['standard', 'minmax'], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--normalize",
    is_flag=True,
    show_default=True,
)
@click.option(
    "--k-best",
    default=0,
    type=click.IntRange(-1, min_open=True),
    show_default=True,
)
@click.option(
    "--save-cfg",
    is_flag=True,
    show_default=True,
)
@click.option(
    "--cfg-path",
    default=STD_CFG_PATH,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def train(
        dataset_path: Path,
        save_model_path: Path,
        random_state: int,
        k_folds: int,
        parallel: bool,
        model: str,
        model_kw: tuple[str, str, str],
        scale: bool,
        scaler: str,
        normalize: bool,
        k_best: int,
        save_cfg: bool,
        cfg_path: Path,
):
    features, target = get_dataset_xy(dataset_path)
    n_jobs = -1 if parallel else None

    try:
        model_kw_fmt = format_kwargs(*model_kw)
        pipeline = create_pipeline(
            model=model,
            random_state=random_state,
            scale=scale,
            scaler=scaler,
            normalize=normalize,
            k_best=k_best,
            n_jobs=n_jobs,
            model_kw=model_kw_fmt
        )
    except Exception as exc:
        raise click.BadParameter(
            f'Raised exception while '
            f'setting pipeline: {exc}'
        )

    with mlflow.start_run():

        kf = KFold(
            n_splits=k_folds,
            shuffle=True,
            random_state=random_state
        )

        accuracy_folds, f1_folds, roc_auc_folds = [], [], []
        accuracy_best, f1_best, roc_auc_best = 0, 0, 0
        for train_index, test_index in kf.split(features):
            x_train, x_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            pipeline.fit(x_train, y_train)

            accuracy, f1, roc_auc_ovr = get_metrics(
                y_test, pipeline.predict(x_test),
                pipeline.predict_proba(x_test))
            accuracy_folds.append(accuracy)
            f1_folds.append(f1)
            roc_auc_folds.append(roc_auc_ovr)

            if accuracy > accuracy_best:
                accuracy_best, f1_best, roc_auc_best = \
                    accuracy, f1, roc_auc_ovr
                dump(pipeline, save_model_path)

        mlflow.log_metrics(
            {'accuracy': float(np.mean(accuracy_folds)),
             'f1_score': float(np.mean(f1_folds)),
             'roc_auc_ovr': float(np.mean(roc_auc_folds))}
        )
        mlflow.sklearn.log_model(pipeline, model)
        mlflow.log_params({
            'model': model, 'random_state': random_state,
            'k_folds': k_folds, 'use_scaler': scale,
            'scaler': scaler, 'normalize': normalize,
            'k_best': k_best
        })
        mlflow.log_param(
            'model_params', 'std' if len(model_kw) == 0 else
            ', '.join(f'{k}={v}' for k, v in model_kw_fmt.items())
        )

        click.echo(
            f'Mean metrics. '
            f'Accuracy: {np.mean(accuracy_folds):.6f}, '
            f'F1 score: {np.mean(f1_folds):.6f}, '
            f'ROC AUC OVR: {np.mean(roc_auc_folds):.6f}'
        )
        click.echo(
            f'Best metrics. '
            f'Accuracy: {accuracy_best:.6f}, '
            f'F1 score: {f1_best:.6f}, '
            f'ROC AUC OVR: {roc_auc_best:.6f}'
        )
        click.echo(f'Best model saved in {save_model_path}')

    if save_cfg:
        save_params_to_cfg(
            dataset_path, save_model_path,
            random_state, k_folds,
            parallel, scale, scaler, normalize,
            k_best, model, model_kw, cfg_path,
        )
        click.echo(f'Train parameters saved in {cfg_path}')


@click.command()
@click.option(
    "-c",
    "--cfg-path",
    default=STD_CFG_PATH,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.pass_context
def train_by_cfg(ctx: click.Context, cfg_path: Path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    kwargs = {}
    try:
        if 'general' in cfg:
            for opt, val in cfg['general'].items():
                if opt in ('dataset_path', 'save_model_path'):
                    kwargs[opt] = Path(val)
                elif opt in ('random_state', 'k_folds', 'k_best'):
                    kwargs[opt] = int(val)
                elif opt in ('scale', 'normalize', 'parallel'):
                    kwargs[opt] = eval(val)
                else:
                    kwargs[opt] = val
        if 'model_kw' in cfg:
            kwargs['model_kw'] = tuple(
                (k, *v.split()) for k, v in
                cfg['model_kw'].items()
            )
    except Exception as exc:
        raise click.BadParameter(
            f'Raised exception while converting '
            f'values from config: {repr(exc)}'
        )
    ctx.invoke(train, **kwargs)


def save_params_to_cfg(
        dataset_path: Path,
        save_model_path: Path,
        random_state: int,
        k_folds: int,
        parallel: bool,
        scale: bool,
        scaler: str,
        normalize: bool,
        k_best: int,
        model: str,
        model_kw: tuple[str, str, str],
        cfg_path: Path,
):
    cfg = configparser.ConfigParser()
    cfg.add_section('general')
    cfg['general']['dataset_path'] = str(dataset_path)
    cfg['general']['save_model_path'] = str(save_model_path)
    cfg['general']['random_state'] = str(random_state)
    cfg['general']['k_folds'] = str(k_folds)
    cfg['general']['parallel'] = str(parallel)
    cfg['general']['scale'] = str(scale)
    cfg['general']['scaler'] = scaler
    cfg['general']['normalize'] = str(normalize)
    cfg['general']['k_best'] = str(k_best)
    cfg['general']['model'] = model

    if len(model_kw) != 0:
        cfg.add_section('model_kw')
        for name, type_, val in model_kw:
            cfg['model_kw'][name] = f'{type_} {val}'

    with open(cfg_path, 'w') as file:
        cfg.write(file)
