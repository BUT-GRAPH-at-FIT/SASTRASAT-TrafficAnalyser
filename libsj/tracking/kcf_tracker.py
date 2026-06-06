__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


from .base_tracker import BaseTracker
import cv2
import numpy as np


class KCFTracker(BaseTracker):
    """IoU tracker whose positions are predicted by per-track OpenCV KCF trackers.

    Each track owns a ``cv2.TrackerKCF`` initialised from its first box; predicted
    positions come from running those visual trackers on the current frame (falling back
    to the last box on failure). More robust to motion than :class:`IoUTracker` at higher
    per-frame cost. Accepts the same constructor arguments as :class:`BaseTracker`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kcf_trackers = {}
    
    def _predict_new_positions(self, frame):
        new_positions = {}
        for track_id, tracker in self._kcf_trackers.items():
            success, (x1,y1,w,h) = tracker.update(frame)
            if success:
                x2 = (x1 + w)/frame.shape[1]
                y2 = (y1 + h)/frame.shape[0]
                x1 /= frame.shape[1]
                y1 /= frame.shape[0]
                new_positions[track_id] = np.asarray([x1, y1, x2, y2])
            else:
                new_positions[track_id] = self._tracks[track_id]["bb"][-1][0:4]
        return new_positions
    
    def _init_new_trackers(self, frame, new_tracks):
        for track_id, track_data in new_tracks.items():
            tracker = cv2.TrackerKCF_create()
            bb = track_data["bb"][0][0:4]
            bb = bb * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            bb[2:4] = bb[2:4] - bb[0:2]
            tracker.init(frame, tuple(bb.astype(int)))
            self._kcf_trackers[track_id] = tracker

    def _delete_terminated_tracks(self, terminated_ids):
        for track_id in terminated_ids:
            del self._kcf_trackers[track_id]