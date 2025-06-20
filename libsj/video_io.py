__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


import logging
from .threading import BaseThread, Event

import av
import numpy as np
from queue import Queue, Full, Empty
import time

# TODO pyav has bug that it returns one frame less than there actually is in the video file
class VideoReader(BaseThread):
    def __init__(self, video_path, frame_step=1, skip_frames=0, take_frames=None, video_ind=0, queue_max_size=128, av_options=None, name="VideoReader"):
        super().__init__(name)
        self.queue = Queue(queue_max_size)
        self.video_ind = video_ind
        self.video_path = video_path
        logging.info("Opening video file: %s"%video_path)
        self.container = av.open(video_path, options=av_options)
        self.frame_step = frame_step
        self.take_frames = take_frames
        if self.take_frames is not None:
            assert self.take_frames > 0
        self.skip_frames = skip_frames
        self._init_stats()


    def _init_stats(self):
        frame_raw = next(self.container.decode(video=self.video_ind)) # read frame to make FPS reliable
        self.container.seek(0)
        frame = np.array(frame_raw.to_image())
        self._frame_shape = frame.shape
        logging.debug("%s duration [s]: %s"%(self.video_path, str(self.duration_time)))
        logging.debug("%s duration [frames]: %s"%(self.video_path, str(self.duration_frames)))
        logging.debug("%s FPS: %s"%(self.video_path, self.fps))
        logging.debug("%s frame shape: %s"%(self.video_path, str(self.frame_shape)))


    @property
    def duration_frames(self):
        return int(self.container.streams.video[self.video_ind].frames)

    @property
    def duration_time(self):
        try:
            return float(self.container.streams.video[self.video_ind].duration * self.container.streams.video[self.video_ind].time_base)
        except TypeError:
            return "Unknown"

    @property
    def fps(self):
        rate = self.container.streams.video[self.video_ind].average_rate
        if (rate == 0) or (rate is None):
            return 0
        else:
            return float(1/rate)

    @property
    def frame_shape(self):
        return self._frame_shape

    def submit(self, item):
        submitted = False
        while not submitted:
            if self.stopped:
                return
            try:
                self.queue.put(item, timeout=1)
                submitted = True
            except Full:
                pass

    def run(self):
        for frame_id, frame_raw in enumerate(self.container.decode(video=0)):
            if self.take_frames is not None and frame_id > self.skip_frames + self.take_frames:
                break
            if frame_id >= self.skip_frames and frame_id % self.frame_step == 0:
                # ts = float(frame_raw.pts * self._time_base) if frame_raw.pts is not None else 0
                ts = int(frame_raw.pts) if frame_raw.pts is not None else 0
                frame = np.array(frame_raw.to_image())
                self.submit((frame_id, frame, ts, frame_raw.is_corrupt))
            if self.stopped:
                return
        self.submit(None)

    def __iter__(self):
        while True:
            data = self.queue.get()
            if data is None:
                break
            yield data




class VideoWriter(BaseThread):
    def __init__(self, video_path, width, height, fps, bit_rate=5120000, codec="libx264", pix_fmt="yuv420p", queue_max_size=128, input_queue=None, swap_channels=False, name="VideoWriter"):
        super().__init__(name)
        if input_queue is None:
            self.queue = Queue(queue_max_size)
        else:
            self.queue = input_queue
        logging.info("Opening output video file: %s"%video_path)
        self.container = av.open(video_path.file, mode="w")
        self.output_stream = self.container.add_stream(codec, rate=fps)
        self.output_stream.width = width
        self.output_stream.height = height
        self.output_stream.pix_fmt = pix_fmt
        self.output_stream.bit_rate = bit_rate
        self.swap_channels = swap_channels
        self.frames_written = 0
        self._finished_event = Event()

    def _set_finished(self):
        logging.debug("Finished")
        self._finished_event.set()

    @property
    def finished(self):
        return self._finished_event.is_set()

    def add_frame(self, frame):
        self.queue.put(frame)

    def _mux_packets(self, packets):
        if packets is not None:
            self.container.mux(packets)
        return packets is not None

    def _encode_frame(self, frame):
        output_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        self._mux_packets(self.output_stream.encode(output_frame))

    def close(self):
        logging.debug("Closing output video file (frames written: %d)"%self.frames_written)
        self.container.close()

    def run(self):
        while not self.stopped:
            try:
                frame = self.queue.get(timeout=1)
                if frame is None:
                    while self._mux_packets(self.output_stream.encode()):
                        pass
                    self._set_finished()
                    return
                if self.swap_channels:
                    frame = np.ascontiguousarray(frame[:, :, ::-1])
                self._encode_frame(frame)
                self.frames_written += 1
            except Empty:
                pass
            except Exception as e: # TODO: Fix exception on the last frame.
                logging.error(f"Error while writing video frame. {str(e)}")
                self._set_finished()
                return

    def __exit__(self, exc_type, value, traceback):
        if exc_type is None:
            self.add_frame(None) # add poison pill to at the end of queue
            logging.debug("Waiting for video writer to finish")
            while not self.finished:
                time.sleep(0.5)
            self.close()
        super().__exit__(exc_type, value, traceback)

