# pixelwise-model

Combobulated by Prof. Dr.-Ing. Mark Schutera and Claude.

Trained model artefacts for [PixelWise](https://github.com/schutera/pixelwise),
the course project from Block 4 of *full stack handwerk*.

This repo separates the **model** (training data, training script, and
serialised artefacts) from the **application** that integrates it. The
application pins a tagged release here and pulls the `.pkl` into
`models/` at deploy time.

## Releases
| Tag    | Artefact                     | Classes    | Notes                |
|--------|------------------------------|------------|----------------------|
| v1.0   | `digit_classifier_v1.pkl`    | 1-9        | Class 0 withheld.    |

Each release ships with its own `MODELCARD.md` describing training data,
capabilities, known failures, intended use, and the pipeline contract.

## Pulling a release into PixelWise

    gh release download v1.0 \
        --repo schutera/pixelwise-model \
        --dir models/

PixelWise's `.env` then pins `MODEL_VERSION=v1.0` and
`MODEL_PATH=models/digit_classifier_v1.pkl`.

## Reproducing v1 locally

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python train.py

This downloads MNIST (~50 MB, cached after the first run), trains a
`Binarizer + LogisticRegression` pipeline on digits 1-9, prints a
classification report, and writes `digit_classifier_v1.pkl` next to
`train.py`.

## Security note
A `.pkl` file deserialises arbitrary Python on load. Only consume
artefacts from this repo's tagged releases (or your own fork). Never
load a `.pkl` from an untrusted source.
