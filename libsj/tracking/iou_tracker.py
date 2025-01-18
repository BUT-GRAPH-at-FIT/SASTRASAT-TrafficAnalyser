__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


from .base_tracker import BaseTracker
import cv2

class IoUTracker(BaseTracker):  
    def _predict_new_positions(self, frame):
        return {track_id: track_data["bb"][-1][0:4] for track_id, track_data in self._tracks.items()}