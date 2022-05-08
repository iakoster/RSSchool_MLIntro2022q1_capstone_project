from pathlib import Path
from typing import Union, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score


def get_metrics(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
):
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average='macro'),
        precision_score(y_true, y_pred, average='macro')
    )


def create_pipeline(
        model: str = 'knn',
        random_state: int = 42,
        n_jobs: int = None,
        model_kw: dict[str, Any] = None,
) -> Pipeline:
    if model_kw is None:
        model_kw = {}

    pipeline_steps = []
    if model == 'knn':
        pipeline_steps.append((
            'knn', KNeighborsClassifier(
                n_jobs=n_jobs, **model_kw)))
    elif model == 'forest':
        pipeline_steps.append((
            'forest', RandomForestClassifier(
                random_state=random_state, n_jobs=n_jobs,
                **model_kw)))
    return Pipeline(steps=pipeline_steps)
