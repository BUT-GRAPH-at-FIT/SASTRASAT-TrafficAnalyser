# Installation Manual

This guide sets up TrafficAnalyser for running the live analysis pipeline. The offline
analysis notebooks need a **separate** environment — see [Notebooks environment](#7-notebooks-environment-separate).

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.11** | Mandatory. The detector loads legacy TensorFlow 1.x frozen graphs through the `tf.compat.v1` API, which only works on the pinned **TensorFlow 2.16** / Python 3.11 combination. Other Python versions are not supported. |
| Linux | The project is developed and run on Linux (`bash`). |
| NVIDIA GPU + CUDA driver | Strongly recommended. Detection and feature extraction run on the GPU; on CPU the pipeline still runs (TensorFlow falls back gracefully) but is far slower. The pinned `nvidia-*` / `tensorrt-libs` wheels target CUDA 11/12. |
| `git`, `pip`, `venv` | Standard tooling. |
| Docker (optional) | Only needed for the Weaviate vector DB used by notebooks 08/09. |

Check your Python version:

```shell
python3.11 --version    # must report 3.11.x
```

## 2. Get the code

```shell
git clone <repository-url> SASTRASAT-TrafficAnalyser
cd SASTRASAT-TrafficAnalyser
```

## 3. Create and activate a virtual environment

```shell
python3.11 -m venv .venv
source .venv/bin/activate
```

Your shell prompt should now be prefixed with `(.venv)`. Re-run `source .venv/bin/activate`
in every new shell before working with the project.

## 4. Install dependencies (order matters)

Install in **exactly** this order. `requirements-tf-libs.txt` pulls `tensorrt-libs` from
NVIDIA's package index and must be installed **before** the main requirements:

```shell
pip install --upgrade pip
pip install -r requirements-tf-libs.txt   # NVIDIA index: tensorrt-libs
pip install -r requirements.txt           # TensorFlow 2.16, OpenCV, PyAV, Hydra, h5py, zmq, ...
```

> Why the order: `requirements-tf-libs.txt` contains
> `--extra-index-url https://pypi.nvidia.com` and `tensorrt-libs`. Installing it first makes
> the TensorRT runtime available before TensorFlow and the rest are resolved.

## 5. Set up `PYTHONPATH`

The project is **not** a pip-installable package — it is a script plus a local `libsj`
library imported by directory. The entry script must be able to find `libsj`, so add the
repository root to `PYTHONPATH` by **sourcing** the helper (do not execute it — `export` only
affects the current shell when sourced):

```shell
source _init_python_path.sh      # prints: Setting PYTHONPATH=/abs/path/to/repo
```

Run this once per shell session (after activating the venv).

## 6. Download the models

Models are **not** included in the repository. Download the archive from the link in
[`models/readme.md`](../models/readme.md):

> <https://nextcloud.fit.vutbr.cz/s/Aa38mZtqorb9DMz>

Unpack it into `models/` so the layout matches
[`models/dir_structure.txt`](../models/dir_structure.txt) — i.e. with `detectors/`,
`classifiers/`, and `feature_extractors/` subfolders. The shipped `config/config.yaml`
expects these four paths by default:

| Config key | Default path |
|------------|--------------|
| `detection.model` | `models/detectors/vehicle_peds_ssd_mobilenet/frozen_inference_graph.pb` |
| `classification.mmr_model` | `models/classifiers/classifier_frontal_ResNet50_2DBB` |
| `classification.color_model` | `models/classifiers/colors_MobileNet` |
| `extractor.model` | `models/feature_extractors/vehicle_MobileNet_AIC_128dim/` |

If you place the models elsewhere, override the paths on the command line (see the
[User Manual](user-manual.md)).

## 7. Notebooks environment (separate)

The Jupyter notebooks under `notebooks/` have dependency conflicts with the main pipeline
(they pull `torch` and `faiss-gpu`). Use a **separate** virtual environment for them:

```shell
python3.11 -m venv .venv-notebooks
source .venv-notebooks/bin/activate
pip install --upgrade pip
pip install -r requirements-notebooks.txt
```

Notebooks 08 and 09 additionally use a Weaviate vector database. Start it with Docker:

```shell
docker compose up -d        # Weaviate on http://localhost:8080 (gRPC on 50051)
```

## 8. Verify the installation

With the main venv active, `PYTHONPATH` set, and models in place, run the pipeline on a
short clip (limiting the frame count keeps the smoke test fast):

```shell
python traffic_analyser.py video.source=/path/to/short_clip.mp4 take_frames=50
```

A successful run prints the resolved config, logs progress, and creates a timestamped
output directory under `outputs/` containing at least `track_meta.csv` and `features.h5`.
See the [User Manual](user-manual.md) for the full output layout and configuration options.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'libsj'`** — `PYTHONPATH` is not set; re-run
  `source _init_python_path.sh` (step 5).
- **`omegaconf.errors.MissingMandatoryValue` / `AttributeError` on `video.source`** —
  `video.source` defaults to `null` and must be supplied (see the User Manual).
- **File-not-found on a model path** — the models were not downloaded/unpacked into `models/`
  (step 6), or the layout does not match `models/dir_structure.txt`.
- **Notebook import errors / version clashes** — you are likely in the main venv; activate
  `.venv-notebooks` instead (step 7).
