__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


from ..threading import ProcessingThread
from queue import Queue, Full, Empty

class TrackerThread(ProcessingThread):
    """Pipeline stage that runs a tracker on each frame's detections.

    Calls the wrapped tracker's ``track`` method and stores the result on the payload
    under ``data["tracks"]``.

    Args:
        tracker: A :class:`~libsj.tracking.base_tracker.BaseTracker` instance.
        input_queue: Queue of detection payloads from the detector stage.
        output_queue: Queue the payload (now with ``tracks``) is forwarded to.
        name: Thread name used in logs and the queue monitor.
    """

    def __init__(self, tracker, input_queue, output_queue, name="Tracker"):
        super().__init__(input_queue, output_queue, name)
        self.tracker = tracker

    def process(self, data):
        """Run the tracker and attach the track dict to ``data["tracks"]``."""
        data["tracks"] = self.tracker.track(data["frame_id"], data["frame"], data["detections"])
        return data
        