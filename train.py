"""Train digit_classifier v1.1: MLPClassifier on MNIST digits 1-9.

A drop-in accuracy upgrade for v1.0. Class scope is unchanged (0 still
withheld), so `classes_` stays ["1".."9"] and the integrating app needs no
code change: it keeps its 9-class CLASSES list. The model swaps
LogisticRegression (~92%) for a small MLP (~98%), shipped as a new artefact
(digit_classifier_v1_1.pkl) alongside the untouched v1.0 file; the app pins
MODEL_PATH to it. The Binarizer step is unchanged so the contract still
matches a frontend that sends canvas pixels binarised at threshold 128.

For the 10-class variant (digits 0-9): ``git checkout v2.0 -- train.py``.

Run:
    python train.py
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer

ARTEFACT = Path(__file__).parent / "digit_classifier_v1_1.pkl"
RANDOM_STATE = 42


def load_mnist_excluding_zero() -> tuple[np.ndarray, np.ndarray]:
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )
    X = X.astype(np.float32) / 255.0
    y = y.astype(str)
    mask = y != "0"
    return X[mask], y[mask]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("binarize", Binarizer(threshold=0.5)),
            (
                "clf",
                # early_stopping stays off: sklearn 1.8.0 scores the
                # validation split with np.isnan on string predictions, which
                # raises. Adam stops on the training-loss plateau (tol /
                # n_iter_no_change) instead; the held-out X_test split below is
                # untouched either way, so the reported accuracy is honest.
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    max_iter=100,
                    n_iter_no_change=8,
                    tol=1e-4,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    print("Fetching MNIST (cached on rerun)...")
    X, y = load_mnist_excluding_zero()
    print(f"Samples: {len(X)}  features: {X.shape[1]}  classes: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_secs = time.perf_counter() - t0

    acc = pipeline.score(X_test, y_test)
    print(f"\nTrained in {train_secs:.1f}s   test accuracy: {acc:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_test, pipeline.predict(X_test), digits=3))

    joblib.dump(pipeline, ARTEFACT)
    print(f"Saved {ARTEFACT.name} ({ARTEFACT.stat().st_size / 1024:.1f} KiB)")
    print(f"classes_: {list(pipeline.classes_)}")
    print(f"n_features_in_: {pipeline.n_features_in_}")


if __name__ == "__main__":
    main()
