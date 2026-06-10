# Programmer Manual

This is the developer-facing reference for TrafficAnalyser: the architecture, how to extend
the pipeline, the public API surface, and the offline tooling. For installation see the
[Installation Manual](installation.md); for operating the tool see the
[User Manual](user-manual.md).

TrafficAnalyser ingests a video file or RTSP stream and runs it through a real-time analysis
pipeline that detects and tracks vehicles and pedestrians, extracts a re-identification
feature vector per vehicle, renders an annotated video, and persists structured per-detection
data. The whole live system is orchestrated from a single script, `traffic_analyser.py`,
which wires together threaded stages from the vendored `libsj` library. A second, **offline**
workflow (notebooks + `matching/` + `tools/`) consumes the data the pipeline emits.

---

## 1. Repository layout

```
traffic_analyser.py     Entry point: defines the pipeline stages and wires them in main().
config/config.yaml      Hydra configuration (all defaults).
libsj/                  Vendored utility library (MIT, 2018):
    threading.py          BaseThread / ProcessingThread + monitor & duplicator threads.
    video_io.py           VideoReader / VideoWriter (PyAV).
    nn/                   ObjectDetector(+Thread), layers, losses, data generators.
    tracking/             BaseTracker, IoUTracker, KCFTracker, TrackerThread.
    plotting.py           OpenCV/matplotlib drawing helpers (cv_draw_text).
    utils.py              IoU, caching, logging, dir helpers.
matching/tools.py       FAISS nearest-neighbour search over re-id features.
tools/embs_tools.py     Read & aggregate features.h5 / track_meta.csv outputs.
notebooks/              Offline re-id / statistics / Weaviate experiments.
models/                 Model files (downloaded separately; see Installation Manual).
docker-compose.yml      Weaviate vector DB for notebooks 08/09.
```

---

## 2. High-level data flow

```
                         video file / RTSP stream
                                   │
                          ┌────────▼────────┐
                          │   VideoReader   │   decode (PyAV), frame skip/step,
                          │   (libsj)       │   per-camera edge cropping
                          └────────┬────────┘
                                   │ frames
                          ┌────────▼────────────┐
                          │ ObjectDetectorThread │  frozen graph (tf.compat.v1)
                          │ (libsj.nn)           │  boxes / scores / classes
                          └────────┬────────────┘   class 1 = vehicle, 2 = pedestrian
                                   │ detections
                          ┌────────▼────────┐
                          │  TrackerThread   │  IoU or KCF tracker;
                          │  (libsj.tracking)│  assigns track_id + status
                          └────────┬────────┘
                                   │ tracks
                          ┌────────▼─────────────┐
                          │ ClassificationThread  │  crop each new vehicle,
                          │                       │  extract normalised 128-d feature
                          └────────┬─────────────┘
                                   │ tracks + features
                          ┌────────▼────────┐
                          │  AnalyseThread   │  position, movement/speed, stats
                          └────────┬────────┘
                                   │
                       ┌───────────▼────────────┐
                       │ QueueDuplicatorThread   │  deep-copy fan-out
                       └────┬───────────────┬────┘
                            │               │
                  ┌─────────▼──────┐  ┌──────▼───────────────┐
                  │  DrawThread    │  │  DataOutputThread     │
                  │  annotate frame│  │  track_meta.csv,      │
                  └────────┬───────┘  │  features.h5, crops,  │
                           │          │  ZMQ PUSH             │
              ┌────────────▼───────┐  └───────────────────────┘
              │ ShowThread (screen)│
              │   or VideoWriter   │
              │   (annotated MP4)  │
              └────────────────────┘
```

The two-way split after `AnalyseThread` lets the visualization branch and the
data-persistence branch run concurrently and independently.

---

## 3. The threading model

Every processing stage is a thread. The contract lives in `libsj/threading.py`.

### `BaseThread`
Wraps `threading.Thread` with a stop `Event`, a `start()`/`stop()`/`exit()` lifecycle, and
context-manager support. `exit()` stops the thread (if alive) and joins it.

### `ProcessingThread(BaseThread)`
The base for every pipeline stage. Subclasses override three hooks:

| Hook | When it runs | Purpose |
|------|--------------|---------|
| `init_thread()` | once, at the start of `run()` | load models, open windows/files — anything that must live on the worker thread |
| `process(data)` | per item | transform the payload and **return** it (returning `None` is only valid when there is no output queue) |
| `finalize_thread()` | once, before exit | flush/close resources |

The base `run()` loop: pull from `input_queue` (1 s timeout), call `process()`, push the
result to `output_queue` (blocking with backpressure if the queue is full), repeat until
stopped or a poison pill arrives.

### Connecting stages
Stages are joined by bounded `queue.Queue` instances (`QUEUE_SIZE = 256`). Each stage gets
its predecessor's output queue as its input queue. `main()` builds the queues and stages,
registers each `(name, queue)` pair with the monitor, then `start()`s everything.

### Two helper threads
- **`QueueDuplicatorThread`** — reads one input queue and **deep-copies** each item to
  multiple output queues. Used to fan the stream out to the draw branch and the data-output
  branch so they don't share mutable state.
- **`QueuesMonitorThread`** — periodically samples every queue's occupancy and logs the mean
  (`report_period=60`s). The bottleneck stage shows up as a consistently full queue.

---

## 4. Shutdown, backpressure, and the payload

These three conventions are the load-bearing parts of the design — preserve them when adding
or reordering stages.

**Poison pill.** When `VideoReader` reaches end-of-stream it pushes `None` down the queue.
Every `ProcessingThread` that receives `None` forwards `None` to its output queue and then
breaks out of its loop. `QueueDuplicatorThread` forwards the pill to *all* its outputs. This
is how end-of-stream propagates cleanly through the whole chain. **A new stage must forward
the poison pill downstream.**

**Backpressure.** Queues are bounded. A producer that finds its output queue full blocks
(with timeout, re-checking `stopped`) rather than dropping data or growing memory. So a slow
consumer throttles the whole upstream chain instead of causing unbounded buffering.

**Lifecycle.** `main()` wraps construction and startup in `try/finally`; the `finally` calls
`t.exit()` on every thread so a crash or interrupt still joins all workers. The main thread
blocks on `data_output.join()` — the terminal sink — to keep the process alive until
processing finishes.

**The payload (`data`).** A single mutable `dict` flows through the pipeline; each stage
enriches it in place and returns the same object. Key fields, by the stage that adds them:

| Field | Added by | Notes |
|-------|----------|-------|
| `frame`, `frame_id` | VideoReader/Detector | decoded RGB frame + index |
| `detections` | Detector | `[x1, y1, x2, y2, class, score, is_front]` rows |
| `tracks` | Tracker | `{track_id: {bb, class, status, score, frame_id, …}}`; `class` 1=vehicle, 2=pedestrian; `status` ∈ {new, detected, undetected, terminated} |
| `feature`, `bb_size`, (`crop`) | Classifier | per-track 128-d L2-normalised vector; crop kept only if `save_crops` |
| `position`, `movement`, `speed`, `is_moving`, `stats` | Analyser | derived geometry + per-frame counts |

> `DrawThread` returns a bare frame (ndarray), not the dict — it is the last stage that needs
> the structured payload.

---

## 5. The stages in detail

**VideoReader** (`libsj/video_io.py`) — decodes with PyAV. Supports `max_fps`, `skip_frames`,
`take_frames`, and per-camera **edge cropping** (`video.line_paddings`) used to mask
irrelevant left/right regions of each fixed camera (see the per-camera crop table in
`config/config.yaml`). RTSP is forced over TCP via `av_options`.

**ObjectDetectorThread** (`libsj/nn/object_detector.py`) — loads a frozen inference graph
through the **`tf.compat.v1`** API (`image_tensor` → `detection_boxes/scores/classes`),
optional input downscaling for speed (`scale_factor`), and a confidence `threshold`. This
reliance on the legacy frozen-graph + `compat.v1` API is why the project is pinned to
TensorFlow 2.16 / Python 3.11.

**TrackerThread** (`libsj/tracking/`) — `IoUTracker` or `KCFTracker`, selected by
`tracker.type`. Associates detections across frames into tracks with stable IDs and a
lifecycle status, terminating tracks after `terminate_after` frames without a detection.

**ClassificationThread** — for each vehicle track in state `new`/`detected`, crops the box
(tight + 15 %-padded "loose" crop), normalises (`(x − 116) / 128`), and runs a Keras feature
extractor (last layer of a MobileNet model) to produce an L2-normalised 128-d embedding stored
on the track. *MMR / colour classifier paths are constructed but their inference is not
currently wired in.*

**AnalyseThread** — computes each track's reference point (vehicle center / pedestrian
bottom-center), short-horizon movement vector and speed (`is_moving` if > 10 px), and
assembles the `stats` dict shown on the status bar. See
[§7 Disabled traffic-analysis logic](#7-disabled-traffic-analysis-logic).

**DrawThread / ShowThread / VideoWriter** — `DrawThread` renders boxes, IDs, movement arrows,
and a status bar onto the frame. The rendered frames then go to `ShowThread` (on-screen
window, when `output.video_output` is null) or `VideoWriter` (annotated MP4).

**DataOutputThread** — the terminal data sink. For each vehicle detection it appends a row to
`track_meta.csv` (record_id, track_id, frame_id, position, confidence, bb_size, crop_path),
accumulates the 128-d features into a resizable `features.h5` dataset, optionally writes crop
JPEGs under `vehicle_crops/<track_id>/`, and connects a ZMQ `PUSH` socket for streaming output.

---

## 6. Extending the pipeline

### Add a processing stage
1. Subclass `ProcessingThread` (`libsj/threading.py`). Use `DrawThread` or `AnalyseThread` in
   `traffic_analyser.py` as templates.
2. Implement the hooks:
   - `init_thread()` — load anything that must live on the worker thread (models, files).
   - `process(data)` — read/enrich the mutable `data` dict and **return it**. Returning
     `None` is only valid for a terminal sink (one whose `output_queue` is `None`); otherwise
     `None` is interpreted as the poison pill.
   - `finalize_thread()` — release resources.
3. Wire it in `main()`: create its output `Queue(QUEUE_SIZE)`, construct the stage with the
   previous stage's output queue as its input, append it to `all_threads`, and append its
   `(name, queue)` pair to `all_queues` so it shows up in the monitor.
4. The base `run()` loop already forwards the poison pill — keep that behaviour if you
   override `run()`.

### Add a tracker
Subclass `BaseTracker` (`libsj/tracking/base_tracker.py`) and override
`_predict_new_positions(frame)` (and optionally `_init_new_trackers` /
`_delete_terminated_tracks`). Then add a branch in `main()` where `cfg.tracker.type` is
matched, mirroring the existing `IoU`/`KCF` cases.

### Use a different detector / extractor
These are config-driven — point `detection.model` / `extractor.model` at a compatible model
(see the [User Manual](user-manual.md)). Changing the *format* (e.g. a SavedModel instead of a
frozen graph) requires editing `ObjectDetector._load_model` in `libsj/nn/object_detector.py`.

---

## 7. Disabled traffic-analysis logic

`AnalyseThread` carries a richer traffic-violation feature set that is **switched off** in the
current configuration. The methods `get_semaphore_state`, `get_violation`, and
`draw_calibration` exist and are documented, but:

- their call sites inside `AnalyseThread.process` are commented out,
- `semaphore` is hardcoded to `"None"` and the per-frame counts are zeroed,
- ROI / protected-area tests and traffic calibration (`self.calib`, sourced from
  `config["calib"]`, currently `None`) are inactive.

Re-enabling them requires uncommenting the relevant blocks **and** supplying the config keys
they expect (`semaphore`, `roi`, `protected_area`, `crossing`) plus a calibration object. The
on-video status bar surfaces these fields, which is why it shows placeholder/zero values.

---

## 8. Configuration (Hydra)

The entry point is `hydra_main`, decorated with
`@hydra.main(config_path="config", config_name="config")`; it injects `config/config.yaml` as
an OmegaConf `DictConfig` into `main()`. There is **no argparse** — override settings on the
command line (`key=value`). The full key reference is in the
[User Manual](user-manual.md#configuration-reference). Each run writes to a timestamped,
per-camera directory derived from the source filename:

```
{output.data_dir}/{camera_name}/YYYY_MM/DD/HH/MM/
    ├── track_meta.csv
    ├── features.h5
    ├── <video_output.file>        # annotated MP4 (if not showing on screen)
    └── vehicle_crops/<track_id>/  # if output.save_crops
```

---

## 9. API reference

Key public classes and functions, by module. Signatures are abbreviated; see the docstrings
in the source for full parameter documentation.

### `traffic_analyser.py` — pipeline stages
| Symbol | Purpose |
|--------|---------|
| `ClassificationThread(model_path, color_model_path, extractor_model_path, in_q, out_q, …)` | Extract and attach a 128-d re-id feature per vehicle track. |
| `AnalyseThread(config, in_q, out_q, calibration_output, …)` | Derive per-track position/movement and build `data["stats"]`. |
| `DrawThread(in_q, out_q, …)` | Render boxes/IDs/arrows + status bar; returns a frame. |
| `ShowThread(in_q, window_name, …)` | Display rendered frames in an OpenCV window (terminal sink). |
| `DataOutputThread(in_q, output_dir, zmq_ip_addr, save_vehicle_crops, …)` | Persist `track_meta.csv`, `features.h5`, crops; ZMQ PUSH (terminal sink). |
| `main(cfg)` | Build, wire, run, and tear down the whole pipeline. |

### `libsj/threading.py` — pipeline primitives
| Symbol | Purpose |
|--------|---------|
| `BaseThread(name)` | Thread with cooperative stop flag and `start`/`stop`/`exit` lifecycle. |
| `ProcessingThread(in_q, out_q, name)` | Base stage; override `init_thread` / `process` / `finalize_thread`. |
| `QueuesMonitorThread(queues, report_period, …)` | Log mean queue occupancy to find bottlenecks. |
| `QueueDuplicatorThread(in_q, out_queues, …)` | Deep-copy each item to several output queues. |

### `libsj/video_io.py` — I/O
| Symbol | Purpose |
|--------|---------|
| `VideoReader(video_path, max_fps, skip_frames, take_frames, …, line_crops, …)` | Decode + crop frames into the pipeline (source thread). |
| `VideoWriter(video_path, width, height, fps, …, input_queue, swap_channels, …)` | Encode frames from a queue to a video file (sink thread). |

### `libsj/nn/object_detector.py` — detection
| Symbol | Purpose |
|--------|---------|
| `ObjectDetector(model_path, session, …, scale_factor, default_threshold)` | Frozen-graph detector; `detect(frame)` / `detect_multi(frames)`. |
| `ObjectDetectorThread(model_path, in_q, out_q, …, gpu_mem, allow_growth, …)` | Detector pipeline stage producing the `detections` array. |

### `libsj/tracking/` — tracking
| Symbol | Purpose |
|--------|---------|
| `BaseTracker(iou_threshold, terminate_after_frames)` | IoU-association tracker; `track(frame_id, frame, detections)`. |
| `IoUTracker(...)` | Predicts each track stays at its last box (fast). |
| `KCFTracker(...)` | Predicts positions with per-track OpenCV KCF trackers (robust). |
| `TrackerThread(tracker, in_q, out_q, …)` | Tracker pipeline stage; stores `data["tracks"]`. |

### `libsj/plotting.py` & `libsj/utils.py`
| Symbol | Purpose |
|--------|---------|
| `cv_draw_text(frame, text, position, …)` | Draw text with a filled background box, in place. |
| `setup_logging(level)` | Configure thread-aware, timestamped root logging. |
| `ensure_dir(d)` | Create a directory (and parents) if missing. |
| `bb_iou(bb_a, bb_b)` | Intersection-over-union of two boxes (used by the trackers). |

---

## 10. Offline re-identification / matching workflow

Separate from the live pipeline and using a **different environment**
(`requirements-notebooks.txt`; see the
[Installation Manual](installation.md#7-notebooks-environment-separate)):

- **`tools/embs_tools.py`** — reads the pipeline's `features.h5` / `track_meta.csv`:
  - `get_file_embs(path)` → `(track_id, embedding)` pairs from one HDF5 file.
  - `get_detection_to_track_map(path)` / `get_detection_to_bb_size_map(path)` → CSV lookups.
  - `get_embs(dir)` → all embeddings under a directory tree, joined with their metadata.
  - `aggregate_embeddings(embs, fn)` with `bb_weighted_average` / `bb_greedy` → one vector per
    track.
  - `accelerated_cosine_similarity(A, B)` → batched GPU cosine-similarity matrix.
  - `get_crops_for_id(root, track_id)` → load saved crop images for a track.
- **`matching/tools.py`** — `find_matches(query, db, …)`: FAISS nearest-neighbour search
  (`IndexFlatIP` quantizer + `IndexIVFFlat`, optional GPU) returning the top-1 match per query.
- **`notebooks/`** — re-identification experiments, per-video statistics, cross-video
  histograms, FAISS-accelerated matching, and a Weaviate vector-DB demo (notebooks `08`/`09`).
  `docker-compose.yml` brings up the Weaviate instance those notebooks use.

---

## 11. Runtime requirements

Why the version pins matter, in brief (full setup in the [Installation Manual](installation.md)):

- **Python 3.11 + TensorFlow 2.16** are mandatory: the detector loads legacy frozen graphs via
  `tf.compat.v1`, which only works on this combination.
- Models are **not** in the repo — download and unpack into `models/` per
  `models/dir_structure.txt`.
- Dependency install order matters: `requirements-tf-libs.txt` (TensorRT/CUDA libs) before
  `requirements.txt`.
- `libsj` is imported by directory, not installed — the repo root must be on `PYTHONPATH`
  (`source _init_python_path.sh`).
