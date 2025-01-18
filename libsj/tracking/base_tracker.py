__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


import copy
from ..utils import bb_iou
import numpy as np

class BaseTracker:
    def __init__(self, iou_threshold=0.5, terminate_after_frames = 5):
        self.iou_threshold = iou_threshold
        self.terminate_after_frames = terminate_after_frames
        self._tracks = {}
        self._next_track_id = 0
        
        
    def track(self, frame_id, frame, detections):
        new_positions = self._predict_new_positions(frame)
        ###############
        # match tracks
        ###############
        unmatched = [True]*len(detections)
        for track_id, track_data in self._tracks.items():
            new_pos = new_positions[track_id]
            ious = np.asarray([bb_iou(new_pos, d) for d in detections[:, 0:4]])
            ious *= (track_data["class"] == detections[:, 4]) # mask different detection classes
            ious *= np.asarray(unmatched, dtype=int) # mask already matched bbs
            if len(ious) > 0:
                max_ind = np.argmax(ious)
                max_iou = ious[max_ind]
                if max_iou >= self.iou_threshold:
                    track_data["frame_id"].append(frame_id)
                    track_data["bb"].append(detections[max_ind])
                    unmatched[max_ind] = False
        ###############
        # initialize new tracks
        ###############
        new_tracks = {}
        for detection in detections[unmatched, :]:
            new_tracks[self._next_track_id] = {
                "bb": [detection],
                "class": detection[4],
                "frame_id": [frame_id], 
            }
            self._next_track_id += 1
        self._init_new_trackers(frame, new_tracks)
        self._tracks.update(new_tracks)
        ###############
        # update status 
        ###############
        terminated_ids = []
        for track_id, track_data in self._tracks.items():
            last_frame_id = track_data["frame_id"][-1]
            if len(track_data["frame_id"]) == 1 and last_frame_id == frame_id:
                track_data["status"] = "new"
            elif last_frame_id == frame_id:
                track_data["status"] = "detected"
            elif frame_id - self.terminate_after_frames > last_frame_id:
                track_data["status"] = "terminated"
                terminated_ids.append(track_id)
            else:
                track_data["status"] = "undetected"
        return_data = copy.deepcopy(self._tracks) # copy before deletion
        for track_id in terminated_ids:
            del self._tracks[track_id]
        self._delete_terminated_tracks(terminated_ids)
        return return_data    
        
    
    def _predict_new_positions(self, frame):
        raise NotImplementedError
    
    def _init_new_trackers(self, frame, new_tracks):
        pass
    
    def _delete_terminated_tracks(self, terminated_ids):
        pass
    