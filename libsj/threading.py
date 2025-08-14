__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"

from threading import Thread, Event
from queue import Queue, Full, Empty
import logging
import time
import copy


class BaseThread(Thread):
    def __init__(self, name=None):
        super().__init__(name=name)
        self._stop_event = Event()
        self.started = False

    def stop(self):
        logging.debug("Stopping thread %s" % self.name)
        self._stop_event.set()

    @property
    def stopped(self):
        return self._stop_event.is_set()

    # start thread
    def start(self):
        self.started = True
        logging.debug("Starting thread %s" % self.name)
        super().start()

    # terminates thread
    def exit(self):
        return self.__exit__(None, None, None)

    # to use in with statement
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, value, traceback):
        if self.is_alive():
            self.stop()
        if self.started:
            logging.debug("Will join thread %s" % self.name)
            self.join()


class ProcessingThread(BaseThread):
    def __init__(self, input_queue, output_queue, name=None):
        super().__init__(name)
        self.input_queue = input_queue
        self.output_queue = output_queue

    def init_thread(self):
        """
        Will be called in the run function
        """
        pass

    def process(self, data):
        """
        Should return the thread output data.
        Do NOT return None if it should be forwarder.
        It is ok to return None only if the output_queue is None.
        """
        raise NotImplementedError

    def finalize_thread(self):
        """
        Will be called before termination of the thread.
        The termination can be either caused by stopping thread or by recieving the poison pill.
        """
        pass

    def run(self):
        self.init_thread()
        while not self.stopped:
            try:
                data = self.input_queue.get(timeout=1)
                if data is None:
                    if self.output_queue is not None:
                        self.output_queue.put(None)  # forward poison pill
                    break
                output_data = self.process(data)
                if self.output_queue is not None:
                    submitted = False
                    while not submitted and not self.stopped:
                        try:
                            self.output_queue.put(output_data, timeout=1)
                            submitted = True
                        except Full:
                            pass
            except Empty:
                pass
        self.finalize_thread()


class QueuesMonitorThread(BaseThread):
    def __init__(self, queues, report_period=120, sampling_period=1, name="QueuesMonitor"):
        super().__init__(name)
        self.queues = queues
        self.report_period = report_period
        self.sampling_period = sampling_period
        self._reset_stats()

    def _reset_stats(self, ):
        self.samples_num = 0
        self.samples_sum = [0.0] * len(self.queues)
        self.last_report = time.time()

    def _sample(self):
        for i in range(len(self.samples_sum)):
            self.samples_sum[i] += self.queues[i][1].qsize()
        self.samples_num += 1

    def _report(self):
        logging.debug("#" * 30)
        for (name, queue), occupancy_sum in zip(self.queues, self.samples_sum):
            if self.samples_num > 0:
                logging.debug("Queue %s mean occupancy: %.1f/%d" % (name, occupancy_sum / self.samples_num, queue.maxsize))
        self._reset_stats()

    def run(self):
        while not self.stopped:
            time.sleep(self.sampling_period)
            self._sample()
            if time.time() - self.report_period > self.last_report:
                self._report()
        self._report()


class QueueDuplicatorThread(BaseThread):
    def __init__(self, input_queue, output_queues, name="QueueDuplicator"):
        super().__init__(name)
        self.input_queue = input_queue
        self.output_queues = output_queues

    def run(self):
        while not self.stopped:
            try:
                data = self.input_queue.get(timeout=1)
                if data is None:
                    for queue in self.output_queues:
                        queue.put(None)  # forward poison pill
                    break
                for queue in self.output_queues:
                    output_data = copy.deepcopy(data)
                    submitted = False
                    while not submitted and not self.stopped:
                        try:
                            queue.put(output_data, timeout=1)
                            submitted = True
                        except Full:
                            pass
            except Empty:
                pass
