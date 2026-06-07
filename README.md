# pixelwise-model

Combobulated by Prof. Dr.-Ing. Mark Schutera and Claude.

Trained model artefacts for [PixelWise](https://github.com/schutera/pixelwise),
the course project from Block 4 of *full stack handwerk*.

This repo separates the **model** (training data, training script, and
serialised artefacts) from the **application** that integrates it. The
application pins a tagged release here and pulls the `.pkl` into
`models/` at deploy time.

## Releases
| Tag    | Artefact                     | Classes    | Notes                          |
|--------|------------------------------|------------|--------------------------------|
| v1.0   | `digit_classifier_v1.pkl`    | 1-9        | Class 0 withheld. LogReg, ~92%.|
| v2.0   | `digit_classifier_v2.pkl`    | 0-9        | Adds class 0. MLP, ~98%.       |

Each release ships with its own `MODELCARD.md` describing training data,
capabilities, known failures, intended use, and the pipeline contract.

## Pulling a release into PixelWise

    gh release download v1.0 \
        --repo schutera/pixelwise-model \
        --dir models/

PixelWise's `.env` then pins `MODEL_VERSION=v1.0` and
`MODEL_PATH=models/digit_classifier_v1.pkl`.

## Reproducing locally

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python train.py

This downloads MNIST (~50 MB, cached after the first run), trains a
`Binarizer + MLPClassifier` pipeline on digits 0-9, prints a
classification report, and writes `digit_classifier_v2.pkl` next to
`train.py`.

To reproduce v1 (`Binarizer + LogisticRegression`, digits 1-9) instead,
check out the v1 training script first: `git checkout v1.0 -- train.py`.

## Security note
A `.pkl` file deserialises arbitrary Python on load. Only consume
artefacts from this repo's tagged releases (or your own fork). Never
load a `.pkl` from an untrusted source.
