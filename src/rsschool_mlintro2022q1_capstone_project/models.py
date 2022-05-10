from typing import Union, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def get_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_pred_proba: Union[pd.Series, np.ndarray],
) -> tuple[float, float, float]:
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
        roc_auc_score(y_true, y_pred_proba, multi_class="ovr"),
    )


def create_pipeline(
    model: str = "knn",
    scale: bool = True,
    scaler: str = "standard",
    normalize: bool = True,
    k_best: int = 0,
    random_state: int = 42,
    n_jobs: int = None,
    model_kw: dict[str, Any] = None,
) -> Pipeline:
    if model_kw is None:
        model_kw = {}
    pipeline_steps = []

    if scale and scaler == "standard":
        pipeline_steps.append(("scaler", StandardScaler()))
    elif scale and scaler == "minmax":
        pipeline_steps.append(("scaler", MinMaxScaler()))

    if normalize:
        pipeline_steps.append(("normalizer", Normalizer()))

    if k_best > 0:
        pipeline_steps.append(("k_best", SelectKBest(k=k_best)))

    if model == "knn":
        pipeline_steps.append(
            ("knn", KNeighborsClassifier(n_jobs=n_jobs, **model_kw))
        )
    elif model == "forest":
        pipeline_steps.append(
            (
                "forest",
                RandomForestClassifier(
                    random_state=random_state, n_jobs=n_jobs, **model_kw
                ),
            )
        )
    return Pipeline(steps=pipeline_steps)
