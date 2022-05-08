from pathlib import Path
from typing import Union

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
        knn_neighbors: int = 5,
        forest_n_estimators: int = 100,
        forest_criterion: str = 'gini',
        forest_max_depth: int = None
) -> Pipeline:

    pipeline_steps = []
    if model == 'knn':
        pipeline_steps.append((
            'knn', KNeighborsClassifier(
                n_neighbors=knn_neighbors
            )))
    elif model == 'forest':
        pipeline_steps.append((
            'forest', RandomForestClassifier(
                n_estimators=forest_n_estimators,
                criterion=forest_criterion,
                max_depth=forest_max_depth,
                random_state=random_state
            )))
    return Pipeline(steps=pipeline_steps)
