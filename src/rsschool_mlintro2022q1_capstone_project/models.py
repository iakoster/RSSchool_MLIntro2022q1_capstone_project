from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier


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
