# TrafficAnalyser

| Basic Info      |      |
|-----------------|------|
| **Project**     | VB02000081 - Bezpečné dopravní systémy s pokročilou technologií |
| **Funding**     | MV SECTECH II |
| **Deliverable** | VB02000081-V3 |
| **Title (EN)**  | Data processing methods for transport purposes |
| **Title (CZ)**  | Metody zpracování dat pro dopravní účely |
| **Authors**     | ŠPAŇHEL, J.; BERAN, V.; HEROUT, A.; ZEMČÍK, P. |
| **Affiliation** | Graph@FIT, Brno University of Technology |

TrafficAnalyser processes a video file or RTSP stream through a real-time pipeline that
**detects** and **tracks** vehicles and pedestrians, **extracts** a 128-dimensional
re-identification feature per vehicle, renders an **annotated video**, and writes
**structured per-detection data** (CSV + HDF5, optional crops). The live pipeline is a single
script, `traffic_analyser.py`, built on the vendored `libsj` library; a separate set of
notebooks performs offline re-identification and matching on the data it produces.

> **Requirements:** Python **3.11** and **TensorFlow 2.16**. The detector loads legacy frozen
> inference graphs through the `tf.compat.v1` API, which only works on this combination. An
> NVIDIA GPU is strongly recommended.

## Quickstart

```shell
# 1. Virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Dependencies — install in THIS order (NVIDIA libs first)
pip install --upgrade pip
pip install -r requirements-tf-libs.txt
pip install -r requirements.txt

# 3. Make the local `libsj` library importable (must be SOURCED)
source _init_python_path.sh

# 4. Download the models (see models/readme.md) into models/

# 5. Run
python traffic_analyser.py video.source=/path/to/video.mp4
```

Configuration is managed by [Hydra](https://hydra.cc); override any setting on the command
line, e.g. `video.max_fps=30 detection.threshold=0.5`. `video.source` is required.

> The Jupyter notebooks in `notebooks/` need a **separate** environment due to dependency
> conflicts — install `requirements-notebooks.txt` in another virtualenv. See the Installation
> Manual.

## Documentation

| Document | For |
|----------|-----|
| [Installation Manual](docs/installation.md) | Setting up the environment, dependencies, models, and notebooks. |
| [User Manual](docs/user-manual.md) | Running analyses, configuration reference, and outputs. |
| [Programmer Manual](docs/programmer-manual.md) | Architecture, extending the pipeline, API reference, and offline tooling. |
