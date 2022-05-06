from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier


def make_pipeline() -> Pipeline:
    return Pipeline([
        ('knn', KNeighborsClassifier())
    ])
