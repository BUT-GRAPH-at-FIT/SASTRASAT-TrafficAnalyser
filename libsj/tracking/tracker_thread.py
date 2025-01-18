__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


from ..threading import ProcessingThread
from queue import Queue, Full, Empty

class TrackerThread(ProcessingThread):
    def __init__(self, tracker, input_queue, output_queue, name="Tracker"):
        super().__init__(input_queue, output_queue, name)
        self.tracker = tracker
        
    def process(self, data):
        data["tracks"] = self.tracker.track(data["frame_id"], data["frame"], data["detections"])
        return data
        