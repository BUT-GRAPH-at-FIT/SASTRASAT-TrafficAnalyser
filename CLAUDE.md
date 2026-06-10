# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TrafficAnalyser processes a video file or RTSP stream through a real-time vehicle/pedestrian
analysis pipeline: detection → tracking → re-identification feature extraction → traffic
analysis → annotated video + structured data output. The whole pipeline lives in a single
entry script (`traffic_analyser.py`) that wires together threaded stages from the vendored
`libsj` library.

The `notebooks/`, `matching/`, and `tools/` directories are a *separate*, offline workflow for
analysing the features the pipeline emits (car re-identification, gallery matching with
faiss/Weaviate). They are not part of the live pipeline and use a different environment.

## Environment & setup

**Python 3.11 is mandatory** — the object detector loads TensorFlow 1.x frozen graphs via
`tf.compat.v1`, which only works on the pinned TF 2.16 / Python 3.11 combination.

Install order matters (TensorRT/CUDA libs must come first):

```shell
pip install --upgrade pip
pip install -r requirements-tf-libs.txt
pip install -r requirements.txt
```

The notebooks have conflicting dependencies (e.g. `torch`, `weaviate`) — use a **separate**
environment with `requirements-notebooks.txt` for anything under `notebooks/`.

Models are not in the repo. Download them (`models/download_models.sh`, or the Nextcloud link
in `models/readme.md`) and unpack into `models/` matching `models/dir_structure.txt`. The
default config expects `models/detectors/`, `models/classifiers/`, `models/feature_extractors/`.

## Running

Configuration is **Hydra-based** (`config/config.yaml`), not argparse. Override config values
on the command line:

```shell
python traffic_analyser.py video.source=/path/to/video.mp4 output.data_dir=outputs
```

`video.source=null` must be set to a real path or RTSP URL. With `output.video_output=null`
the pipeline opens an on-screen window (`ShowThread`); otherwise it writes an annotated MP4.

There is no test suite. `test_hydra.py` is a minimal Hydra smoke test, not a unit test.

## Architecture

See the **[Programmer Manual](docs/programmer-manual.md)** for the full system overview
(architecture, extension guide, API reference). The essentials:

- The live pipeline is a chain of threads connected by bounded queues. `main()` in
  `traffic_analyser.py` builds the stages (Reader → Detector → Tracker → Classifier → Analyser
  → fan-out → Draw + DataOutput), each subclassing `ProcessingThread` (`libsj/threading.py`)
  and implementing `init_thread()` / `process(data)` / `finalize_thread()`.
- A single **mutable `data` dict** flows through the pipeline; each stage enriches it in place
  rather than building new payloads.
- Three conventions are load-bearing — preserve them when adding/reordering stages:
  **poison-pill** (`None`) propagation for clean end-of-stream shutdown, **bounded-queue
  backpressure** (`QUEUE_SIZE = 256`), and the `try/finally` + `t.exit()` lifecycle in `main()`.
- **Much analysis logic is disabled**: `get_semaphore_state`, `get_violation`,
  `draw_calibration`, ROI/protected-area checks, and traffic calibration are present but
  commented out (`semaphore` is hardcoded to `"None"`, stats are zeroed). This is a
  feature-extraction / re-id configuration of a richer traffic-violation system — don't assume
  violation detection works without re-enabling it and wiring up the config keys it expects
  (`semaphore`, `roi`, `protected_area`, `crossing`).
- `libsj` is a vendored utility library (threading primitives, detector/tracker wrappers, video
  I/O, plotting). Treat it as a stable dependency; new pipeline behaviour generally belongs in
  `traffic_analyser.py` as a new `ProcessingThread`.

## Output layout

Per run, output goes to a timestamped path:
`{output.data_dir}/{camera_name}/YYYY_MM/DD/HH/MM/`, containing `track_meta.csv`,
`features.h5`, the annotated video, and (if `output.save_crops`) `vehicle_crops/<track_id>/`.
`camera_name` is derived from the video source filename.
