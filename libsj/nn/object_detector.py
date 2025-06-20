__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


import tensorflow as tf
import numpy as np
import logging
import cv2
from ..threading import ProcessingThread


class ObjectDetector:
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
        assert frame.ndim == 3
        # speed up detection by scaling
        if self.scale_factor is not None:
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
        frame_id, frame, frame_ts, frame_corrupted = data # assume input from video reader
        boxes, classes, scores = self.detector.detect(frame, return_scores=True)
        data = {
            "frame_id": frame_id,
            "frame_ts": frame_ts,
            "frame": frame,
            "is_corrupted": frame_corrupted,
            "detections": np.hstack((boxes, classes.reshape(-1, 1), scores.reshape(-1, 1)))
        }
        return data
    
    def finalize_thread(self):
        logging.debug("Closing TF session")
        self.session.close()
    