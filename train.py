"""Train digit_classifier_v1: LogisticRegression on MNIST digits 1-9.

The artefact this script produces is the contract that pixelwise/app/classifier.py
consumes. Class 0 is intentionally withheld so a v2 release can add it later
without collecting any new data.

Run:
    python train.py
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer

ARTEFACT = Path(__file__).parent / "digit_classifier_v1.pkl"
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
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
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
