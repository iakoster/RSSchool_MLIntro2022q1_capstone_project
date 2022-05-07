from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
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
        knn_neighbors: int = 5,
) -> Pipeline:
    assert model in (
        'knn',
    ), 'invalid model: %s' % model

    pipeline_steps = []
    if model == 'knn':
        pipeline_steps.append(('knn', KNeighborsClassifier(
            n_neighbors=knn_neighbors
        )))
    return Pipeline(steps=pipeline_steps)
