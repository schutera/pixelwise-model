# MODELCARD.md: digit_classifier v1.1

## Training Data
MNIST digits 1-9, ~57k train / ~6.3k test, public domain.
Class 0 withheld, same scope as v1.0; a v2 release adds it.
Pixel values normalised from [0, 255] to [0, 1], then binarised at 0.5
before training.

## Capabilities
Predict handwritten digits 1-9.
Expected accuracy: ~98% (MLP, 0.9800 on the held-out 6.3k test split),
up from ~92% for the v1.0 LogisticRegression baseline.

## Known Failures
8 is the weakest class (f1 0.972), with the residual 6/9 and 3/8
confusions from v1.0 much reduced. Messy, rotated, or heavily skewed
digits still fail. Class 0 is out of distribution and withheld;
predictions on a hand-drawn 0 are arbitrary and should not be trusted.

## Intended Use
28x28 canvas drawings, greyscale, pixel values in [0, 255] (uint8) or
[0, 1] (float). Out of scope: photos, non-digit characters, rotated or
heavily skewed inputs.

## Pipeline Contract
The artefact is an `sklearn.pipeline.Pipeline` with two named steps:

- `binarize`: `Binarizer(threshold=0.5)` — idempotent on already-binarised
  {0.0, 1.0} inputs, so callers that pre-binarise at uint8 threshold 128
  pass through unchanged.
- `clf`: `MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
  solver="adam")`.

Inputs:
- shape `(N, 784)` (flattened 28x28),
- dtype `float32` or `float64`,
- value range either {0.0, 1.0} (pre-binarised) or [0, 1] (the pipeline
  binarises at 0.5).

`pipeline.classes_` returns `["1", "2", "3", "4", "5", "6", "7", "8", "9"]`,
unchanged from v1.0, so the integrating app needs no code change.
`pipeline.n_features_in_` returns `784`.

## Reproducibility
Trained by `train.py` in this repo with `random_state=42` and a 90/10
stratified split. `early_stopping` is off: scikit-learn 1.8.0 scores the
validation split with `np.isnan` on string predictions and raises, so
Adam stops on the training-loss plateau instead. Rerun:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python train.py

MNIST downloads automatically via `sklearn.datasets.fetch_openml` and
is cached in `~/scikit_learn_data/`.

## Version
v1.1 — drop-in accuracy upgrade over v1.0: same 9-class scope, swaps the
LogisticRegression baseline for an MLP (~98%). Ships as a new artefact
`digit_classifier_v1_1.pkl` alongside the untouched v1.0 file; the app
pins `MODEL_VERSION=v1.1` and `MODEL_PATH=models/digit_classifier_v1_1.pkl`.
