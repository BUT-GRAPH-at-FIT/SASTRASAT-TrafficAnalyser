__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"


import logging
from .threading import BaseThread, Event

import av
import numpy as np
from queue import Queue, Full, Empty
import time

crop_dict = {
    #                 left line (front)  ,   right line (back)
    #               (l  , r   , t  , b  ),
    "JS-BM-P487":  ((450, 850 , 650, 20 ), (1100, 20 , 650, 20 )),
    "JK-CE-P571":  ((                   ), (20  , 20 , 150, 20 )),
    "JS-BM-P488":  ((20 , 900 , 220, 130), (1100, 20 , 220, 130)),
    "JS-BM-P489":  ((20 , 750 , 300, 130), (1220, 20 , 300, 130)),
    "JS-BM-P490":  ((20 , 970 , 380, 20 ), (1030, 150, 380, 20 )),
    "JS-BM-P491":  ((20 , 1100, 250, 20 ), (880 , 250, 250, 20 )),
    "JS-BM-P493":  ((20 , 890 , 230, 20 ), (1050, 150, 200, 20 )),
    "JS-BM-P4951": ((20 , 390 , 120, 20 ), (350 , 20 , 120, 20 )),
    "JS-BM-P498":  ((20 , 800 , 300, 130), (1190, 20 , 300, 130)),
    "JS-CM-P492":  ((20 , 950 , 250, 130), (1090, 20 , 250, 130)),
    "JU-PO-P573":  ((20 , 340 , 150, 20 ), (415 , 60 , 150, 20 )),
    "PJ-CE-P551":  ((20 , 370 , 300, 20 ), (400 , 20 , 300, 20 )),
    "SD-PO-P576":  ((20 , 335 , 200, 20 ), (405 , 20 , 200, 20 )),
    "SG-CE-P574":  ((20 , 385 , 135, 20 ), (365 , 20 , 135, 20 )),
    "SU-CE-P575":  ((20 , 385 , 150, 20 ), (360 , 20 , 150, 20 )),
}

def _select_crop(video_path) -> tuple:
    for key in crop_dict.keys():
        if video_path.split("/")[-1].startswith(key):
            return crop_dict[key]

    return (20, 20, 20, 20), (20, 20, 20, 20)

# TODO: Support both top/bottom padding based on crop
def _pad_black(frame, pad):
    return np.pad(
        frame,
        ((0, pad), (0, 0), (0, 0)),
        mode='constant',
        constant_values=0
    )

def _crop_frame(frame, crop):
    combined_frame = None
    left_width = -1

    for line in crop:
        if len(line) <= 0:
            left_width = 0
            continue

        cropped_line = frame[
            line[2]:-line[3],
            line[0]:-line[1],
        ]

        if combined_frame is None:
            combined_frame = cropped_line

            if left_width < 0:
                left_width = combined_frame.shape[1]
        else:
            # TODO: Support both top/bottom padding based on crop
            if combined_frame.shape[0] < cropped_line.shape[0]:
                combined_frame = _pad_black(combined_frame, cropped_line.shape[0] - combined_frame.shape[0])
            elif combined_frame.shape[0] > cropped_line.shape[0]:
                cropped_line = _pad_black(cropped_line, combined_frame.shape[0] - cropped_line.shape[0])

            combined_frame = np.concatenate((combined_frame, cropped_line), axis=1)

    return combined_frame, left_width


# TODO pyav has bug that it returns one frame less than there actually is in the video file
class VideoReader(BaseThread):
    def __init__(self, video_path, max_fps=25, skip_frames=0, take_frames=None, video_ind=0, queue_max_size=128, av_options=None, name="VideoReader"):
        super().__init__(name)
        self.queue = Queue(queue_max_size)
        self.video_ind = video_ind
        self.video_path = video_path
        self.crop = _select_crop(video_path)
        logging.info("Opening video file: %s"%video_path)
        self.container = av.open(video_path, options=av_options)
        self.frame_step = max(1, int(round((1 / self.fps) / max_fps))) if max_fps > 0 else 1
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
                orig_frame = np.array(frame_raw.to_image())

                # crop the best camera view
                frame, separator = _crop_frame(orig_frame, self.crop)

                # Capture them all!
                fw, fh = frame.shape[1], frame.shape[0]
                ofw, ofh = orig_frame.shape[1], orig_frame.shape[0]

                def transform_fn(x1, y1, x2, y2):
                    is_left = lambda x: x < (separator / ofw)
                    line_idx = lambda x: 0 if is_left(x) else 1
                    x_px = lambda x: x * fw

                    # TODO: Simplify
                    return [
                        (self.crop[line_idx(x1)][0] + (x_px(x1) if is_left(x1) else x_px(x1) - separator)) / ofw,
                        (self.crop[line_idx(x1)][2] + y1 * fh) / ofh,
                        (self.crop[line_idx(x1)][0] + (x_px(x2) if is_left(x1) else x_px(x2) - separator)) / ofw,
                        (self.crop[line_idx(x1)][2] + y2 * fh) / ofh,
                    ]

                self.submit((frame_id, orig_frame, frame, ts, frame_raw.is_corrupt, separator, transform_fn))
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
        self.container = av.open(video_path, mode="w")
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
        # In newer version of AV, encode() does return empty array instead of None.
        return packets is not None and len(packets) > 0

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

    def __exit__(self, exc_type, value, traceback):
        if exc_type is None:
            self.add_frame(None) # add poison pill to at the end of queue
            logging.debug("Waiting for video writer to finish")
            while not self.finished:
                time.sleep(0.5)
            self.close()
        super().__exit__(exc_type, value, traceback)

