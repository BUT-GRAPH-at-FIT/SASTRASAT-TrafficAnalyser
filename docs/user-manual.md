# User Manual

This manual is for operators running traffic analyses. For setup see the
[Installation Manual](installation.md); for internals and extending the code see the
[Programmer Manual](programmer-manual.md).

## What it does

TrafficAnalyser ingests a video file or RTSP stream and runs each frame through a pipeline
that **detects** vehicles and pedestrians, **tracks** them across frames, **extracts** a
128-dimensional re-identification feature for each vehicle, **annotates** the video, and
**writes** structured per-detection data (CSV + HDF5, optional crops). Detected objects are
labelled by class: **1 = vehicle, 2 = pedestrian**.

## Running an analysis

Activate the environment, set `PYTHONPATH`, then launch the script with at least a video
source:

```shell
source .venv/bin/activate
source _init_python_path.sh
python traffic_analyser.py video.source=/path/to/video.mp4
```

`video.source` is **required** — it defaults to `null` and the run fails without it. It can
be a local file path or an RTSP/stream URL.

### Overriding configuration

Configuration is managed by [Hydra](https://hydra.cc). All defaults live in
`config/config.yaml`; override any value on the command line as `key=value`, using dots for
nested keys. Multiple overrides are space-separated:

```shell
python traffic_analyser.py \
    video.source=rtsp://camera/stream \
    video.max_fps=30 \
    detection.threshold=0.5 \
    tracker.type=KCF \
    output.save_crops=false
```

To preview the fully-resolved configuration without running, append `--cfg job`.

## Common scenarios

**Analyse a local file:**
```shell
python traffic_analyser.py video.source=/data/clips/junction.mp4
```

**Analyse an RTSP stream** (RTSP-over-TCP options are applied automatically):
```shell
python traffic_analyser.py video.source=rtsp://user:pass@host:554/stream
```

**Preview on screen instead of writing a video** — set the video output to null; an OpenCV
window opens (`ShowThread`):
```shell
python traffic_analyser.py video.source=clip.mp4 output.video_output=null
```

**Write an annotated MP4** (the default) — leave `output.video_output` as configured; the
file is written into the run's timestamped output directory.

**Process only part of a video** — skip leading frames and/or cap the count:
```shell
python traffic_analyser.py video.source=clip.mp4 video.skip_frames=500 video.take_frames=1000
```

**Disable saving vehicle crop images** (smaller output):
```shell
python traffic_analyser.py video.source=clip.mp4 output.save_crops=false
```

**Use different models:**
```shell
python traffic_analyser.py video.source=clip.mp4 \
    detection.model=models/detectors/vehicle_peds_frcnn_res50/frozen_inference_graph.pb
```

## Configuration reference

All keys below are in `config/config.yaml`. Defaults are shown; override any with
`group.key=value`.

### `app`
| Key | Default | Meaning |
|-----|---------|---------|
| `app.monitor_report_period` | `60` | Seconds between queue-occupancy log reports. |
| `app.queue_size` | `256` | Capacity of the inter-stage queues. |
| `app.data_output_api` | `null` | Optional URL of an external data API. |
| `app.av_options` | `{rtsp_transport: tcp, buffer_size: 2048, prefer_tcp: 1}` | Options passed to the video decoder (mainly for RTSP). |

### `video`
| Key | Default | Meaning |
|-----|---------|---------|
| `video.source` | `null` | **Required.** Video file path or RTSP/stream URL. |
| `video.max_fps` | `25` | Target processing frame rate; frames are subsampled to approximate it. |
| `video.skip_frames` | `0` | Number of leading frames to skip. |
| `video.take_frames` | `null` | Max frames to process after skipping (`null` = all). |
| `video.line_paddings.left_line` | `[20, 750, 300, 130]` | Left camera-view crop rectangle `(left, right, top, bottom)` padding in pixels. |
| `video.line_paddings.right_line` | `[1220, 20, 300, 130]` | Right camera-view crop rectangle. |

The reader stitches the left and right crop rectangles into the analysed frame, masking
irrelevant edges of a fixed camera. `config/config.yaml` includes a **per-camera crop table**
(commented) with the correct `left_line`/`right_line` values for each known camera ID
(e.g. `JS-BM-P489*`, `JU-PO-P573*`); copy the matching pair into your config or pass them as
overrides.

### `detection`
| Key | Default | Meaning |
|-----|---------|---------|
| `detection.model` | `models/detectors/vehicle_peds_ssd_mobilenet/frozen_inference_graph.pb` | Path to the frozen detection graph. |
| `detection.gpu_mem` | `0.7` | Fraction of GPU memory reserved for detection. |
| `detection.threshold` | `0.45` | Minimum detection confidence. |
| `detection.scale_factor` | `225` | Input downscale for speed (values > 1 are treated as a target height in pixels). |

### `tracker`
| Key | Default | Meaning |
|-----|---------|---------|
| `tracker.type` | `IoU` | Tracker algorithm: `IoU` (fast) or `KCF` (visual, more robust, slower). |
| `tracker.iou` | `0.45` | IoU threshold for matching detections to tracks. |
| `tracker.terminate_after` | `7` | Frames a track may go undetected before it is terminated. |

### `classification` / `extractor`
| Key | Default | Meaning |
|-----|---------|---------|
| `classification.mmr_model` | `models/classifiers/classifier_frontal_ResNet50_2DBB` | Make/model classifier (loaded; inference not currently wired in). |
| `classification.color_model` | `models/classifiers/colors_MobileNet` | Colour classifier (loaded; inference not currently wired in). |
| `extractor.model` | `models/feature_extractors/vehicle_MobileNet_AIC_128dim/` | Re-identification feature extractor (produces the 128-d vectors). |
| `extractor.batch_size` | `128` | Batch size for feature extraction. |

### `output`
| Key | Default | Meaning |
|-----|---------|---------|
| `output.data_dir` | `outputs` | Base directory for all run output. |
| `output.save_crops` | `True` | If true, save per-detection vehicle crop JPEGs. |
| `output.video_output` | `{file: processed_file.mp4, codec: mpeg4, fps: 15}` | Annotated-video settings. Set `output.video_output=null` to show on screen instead of writing a file. |

## Outputs

Each run writes to a timestamped, per-camera directory. `camera_name` is derived from the
source filename:

```
{output.data_dir}/{camera_name}/YYYY_MM/DD/HH/MM/
    ├── track_meta.csv              # one row per vehicle detection
    ├── features.h5                 # 128-d re-id features + record ids
    ├── processed_file.mp4          # annotated video (unless showing on screen)
    └── vehicle_crops/<track_id>/   # crop JPEGs (only if output.save_crops=true)
```

**`track_meta.csv`** columns: `record_id`, `track_id`, `frame_id`, `position`, `confidence`,
`bb_size`, `crop_path`.

**`features.h5`** datasets: `track_ids` (the per-record identifiers) and `features` (the
matching `N × 128` float feature matrix). The two are aligned by row.

The annotated video shows each track's bounding box, ID, reference point, movement arrows
for moving vehicles, and a top status bar with per-frame counts.

> Note: the on-video status bar includes fields for semaphore state, violations, and crossing
> activity. These belong to a traffic-violation feature set that is **currently disabled** in
> the code, so they display placeholder/zero values. See the Programmer Manual for details.

## Offline analysis (notebooks)

The pipeline's `features.h5` and `track_meta.csv` outputs feed an offline re-identification
and statistics workflow in `notebooks/` (single-video stats, cross-video histograms, car
re-identification, FAISS-accelerated matching, and a Weaviate vector-DB demo). These run in a
**separate** environment (`requirements-notebooks.txt`) — see the
[Installation Manual](installation.md#7-notebooks-environment-separate). The data formats and
helper functions they build on are documented in the [Programmer Manual](programmer-manual.md).
