__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


import tensorflow as tf
import numpy as np
import logging
import cv2
from ..threading import ProcessingThread


class ObjectDetector:
    """Wraps a TensorFlow 1.x frozen-graph detector loaded via ``tf.compat.v1``.

    Loads a frozen inference graph into the given session and exposes
    :meth:`detect`/:meth:`detect_multi`, which return normalised boxes (reordered to
    ``[x1, y1, x2, y2]``) plus class ids and optional scores, filtered by a confidence
    threshold.

    Args:
        model_path: Path to the frozen inference graph (``.pb``).
        session: A ``tf.compat.v1.Session`` whose graph the model is imported into.
        prefix: Name scope used when importing the graph.
        scale_factor: Optional input downscale for speed; a value > 1 is treated as a
            target height in pixels and converted to a ratio on first use.
        default_threshold: Confidence threshold used when ``detect`` is called without one.
    """

    def __init__(self, model_path, session, prefix="detector", scale_factor=None, default_threshold=0.5):
        self.model_path = model_path
        self.session = session
        self.default_threshold = default_threshold
        self.scale_factor = scale_factor
        self.prefix = prefix
        self._load_model()

    def _load_model(self):
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name=self.prefix)
        self.image_tensor = self._get_tensor('image_tensor:0')
        self.detection_boxes = self._get_tensor('detection_boxes:0')
        self.detection_scores = self._get_tensor('detection_scores:0')
        self.detection_classes = self._get_tensor('detection_classes:0')
        
    def _get_tensor(self, name):
        # if self.prefix is not None and self.prefix != "":
        #     name = self.prefix + "/" + name
        return self.session.graph.get_tensor_by_name(name)
    
    def _threshold(self, arrays, threshold, scores_arrays_ind=1):
        mask = arrays[scores_arrays_ind] >= threshold
        return [arr[mask] for arr in arrays]
                
    def _rearange_boxes(self, boxes):
        return boxes[:, [1,0,3,2]]
        
    def detect(self, frame, return_scores = False, threshold = None):
        """Run detection on a single frame.

        Args:
            frame: HxWx3 image array.
            return_scores: If True, also return per-detection scores.
            threshold: Confidence threshold; falls back to ``default_threshold`` if ``None``.

        Returns:
            ``(boxes, classes)`` or ``(boxes, classes, scores)`` if ``return_scores`` is
            True. Boxes are normalised ``[x1, y1, x2, y2]``.
        """
        assert frame.ndim == 3
        # speed up detection by scaling
        if self.scale_factor is not None:
            if self.scale_factor > 1:
                self.scale_factor = self.scale_factor / frame.shape[0]

            frame = cv2.resize(frame,(0,0),fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_AREA)
        if threshold is None:
            threshold = self.default_threshold
        boxes, scores, classes = self.session.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                                  feed_dict={self.image_tensor: frame[None, ...]})
        
        boxes, scores, classes = self._threshold((boxes[0], scores[0], classes[0]), threshold)
        boxes = self._rearange_boxes(boxes)
        if return_scores:
            return boxes, classes, scores
        else:
            return boxes, classes
            
    def detect_multi(self, frames, return_scores = False, threshold = None):
        """Run detection on a batch of frames.

        Args:
            frames: NxHxWx3 batch of images.
            return_scores: If True, include per-detection scores in each result.
            threshold: Confidence threshold; falls back to ``default_threshold`` if ``None``.

        Returns:
            A list with one ``(boxes, classes[, scores])`` tuple per input frame.
        """
        assert frames.ndim == 4
        if threshold is None:
            threshold = self.default_threshold
        data = self.session.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                 feed_dict={self.image_tensor: frames})
        
        result_data = []
        for boxes, scores, classes in zip(*data):
            boxes, scores, classes = self._threshold((boxes, scores, classes), threshold)
            boxes = self._rearange_boxes(boxes)
            if return_scores:
                result_data.append((boxes, classes, scores))
            else:
                result_data.append((boxes, classes,))
        return result_data
            
        
        
class ObjectDetectorThread(ProcessingThread):
    """Pipeline stage that runs the object detector on each frame from the reader.

    Creates a TF session and :class:`ObjectDetector` on the worker thread, then for each
    frame produces a payload dict with ``frame_id``, ``frame_ts``, ``frame`` (original),
    ``is_corrupted`` and a ``detections`` array of ``[x1, y1, x2, y2, class, score,
    is_front]`` rows, with boxes mapped back to original-frame coordinates.

    Args:
        model_path: Path to the frozen inference graph (``.pb``).
        input_queue: Queue of frame tuples from :class:`~libsj.video_io.VideoReader`.
        output_queue: Queue the detection payload is forwarded to.
        prefix: Name scope used when importing the graph.
        threshold: Detection confidence threshold.
        gpu_mem: Per-process GPU memory fraction for the TF session.
        allow_growth: If True, let the TF session grow GPU memory on demand.
        detector_scale_factor: Optional input downscale passed to :class:`ObjectDetector`.
        name: Thread name used in logs and the queue monitor.
    """

    def __init__(self, model_path, input_queue, output_queue, prefix="detector", threshold=0.5, gpu_mem=0.9, allow_growth=False, detector_scale_factor=None, name="Detector"):
        super().__init__(input_queue, output_queue, name)
        self.model_path = model_path
        self.prefix = prefix
        self.threshold = threshold
        self.gpu_mem = gpu_mem
        self.allow_growth = allow_growth
        self.detector_scale_factor = detector_scale_factor
        
    
    def init_thread(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_mem
        config.gpu_options.allow_growth = self.allow_growth
        self.session = tf.compat.v1.Session(config=config)
        self.detector = ObjectDetector(self.model_path, self.session, scale_factor=self.detector_scale_factor, default_threshold=self.threshold)
        
    
    def process(self, data):
        frame_id, orig_frame, frame, frame_ts, frame_corrupted, separator, transform_fn = data # assume input from video reader
        boxes, classes, scores = self.detector.detect(frame, return_scores=True)

        # TODO: This can be cleaner...
        boxes = np.stack(
            [np.array(transform_fn(x1, y1, x2, y2))
             for x1, y1, x2, y2 in boxes]
        ) if len(boxes) > 0 else np.empty((0, 4))
        is_front = np.asarray([bb[0] >= separator for bb in boxes])

        data = {
            "frame_id": frame_id,
            "frame_ts": frame_ts,
            "frame": orig_frame,
            "is_corrupted": frame_corrupted,
            "detections": np.hstack((boxes, classes.reshape(-1, 1), scores.reshape(-1, 1), is_front.reshape(-1, 1))),
        }
        return data
    
    def finalize_thread(self):
        logging.debug("Closing TF session")
        self.session.close()
    