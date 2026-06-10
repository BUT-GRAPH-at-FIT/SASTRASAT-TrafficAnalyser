__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


from .base_tracker import BaseTracker
import cv2

class IoUTracker(BaseTracker):
    """IoU tracker that predicts each track stays at its last bounding box.

    The simplest motion model: associations rely purely on overlap between detections and
    the previous box. Fast and dependency-free; best for high frame rates or slow motion.
    """

    def _predict_new_positions(self, frame):
        return {track_id: track_data["bb"][-1][0:4] for track_id, track_data in self._tracks.items()}