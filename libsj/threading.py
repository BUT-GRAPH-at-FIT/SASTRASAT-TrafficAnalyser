__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"

from threading import Thread, Event
from queue import Queue, Full, Empty
import logging
import time
import copy


class BaseThread(Thread):
    """A ``threading.Thread`` with a cooperative stop flag and lifecycle helpers.

    Provides a ``stop()``/``stopped`` Event, a ``start()`` that records that the thread was
    launched, an ``exit()`` that stops and joins, and context-manager support so a thread
    can be used in a ``with`` block.

    Args:
        name: Thread name used in logs.
    """

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
    """Base class for a single stage in the queue-connected processing pipeline.

    The ``run()`` loop pulls one item from ``input_queue``, passes it to :meth:`process`,
    and pushes the result to ``output_queue`` (blocking with backpressure if it is full).
    A ``None`` item is a *poison pill*: it is forwarded downstream and ends the loop, so
    end-of-stream propagates cleanly through the whole chain. Subclasses override
    :meth:`init_thread`, :meth:`process` and :meth:`finalize_thread`.

    Args:
        input_queue: Queue this stage consumes from.
        output_queue: Queue this stage forwards results to, or ``None`` for a terminal sink.
        name: Thread name used in logs and the queue monitor.
    """

    def __init__(self, input_queue, output_queue, name=None):
        super().__init__(name)
        self.input_queue = input_queue
        self.output_queue = output_queue

    def init_thread(self):
        """Hook run once at the start of ``run()``, on the worker thread.

        Use it for resources that must live on this thread (model loading, file/socket
        handles, GUI windows). Default implementation does nothing.
        """
        pass

    def process(self, data):
        """Transform one input item and return the output.

        Must return the value to forward downstream. Returning ``None`` is only valid when
        ``output_queue`` is ``None`` (a terminal sink); otherwise ``None`` would be
        mistaken for the poison pill.

        Args:
            data: One item pulled from ``input_queue``.

        Returns:
            The payload to push to ``output_queue``.
        """
        raise NotImplementedError

    def finalize_thread(self):
        """Hook run once before the thread terminates (stop or poison pill).

        Use it to flush/close resources opened in :meth:`init_thread`. Default
        implementation does nothing.
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
    """Background thread that periodically logs mean occupancy of the pipeline queues.

    Samples every registered queue's size each ``sampling_period`` seconds and logs the
    mean occupancy every ``report_period`` seconds. A consistently full queue identifies
    the bottleneck stage.

    Args:
        queues: List of ``(name, queue)`` pairs to monitor.
        report_period: Seconds between logged reports.
        sampling_period: Seconds between occupancy samples.
        name: Thread name used in logs.
    """

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
    """Fan-out thread that deep-copies each input item to several output queues.

    Used to split the pipeline so independent branches (e.g. drawing and data output) do
    not share mutable payload state. The poison pill is forwarded to every output queue.

    Args:
        input_queue: Queue to read from.
        output_queues: Queues each item is deep-copied into.
        name: Thread name used in logs.
    """

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
