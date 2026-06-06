__author__ = "Jakub Sochor"
__copyright__ = "Copyright 2018, Jakub Sochor"
__license__ = "MIT"

import pickle
import os
import numpy as np
import sys
import tqdm
import logging
import time
import random
import subprocess
import platform


# %%
def load_cache(path, encoding="latin-1", fix_imports=True):
    """
    encoding latin-1 is default for Python2 compatibility
    """
    with open(path, "rb") as f:
        return pickle.load(f, encoding=encoding, fix_imports=True)


# %%
def save_cache(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(data, f)


# %%
def save_np_cache(path, data):
    ensure_dir(os.path.dirname(path))
    np.save(path, data)


# %%
def ensure_dir(d):
    """Create directory ``d`` (and parents) if it does not already exist.

    An empty path is a no-op (for compatibility with ``os.path.dirname`` on a bare
    filename), and an existing directory is tolerated without error.

    Args:
        d: Directory path to ensure exists.
    """
    if len(d) == 0:  # for empty dirs (compatibility with os.path.dirname("xxx.yy"))
        return
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError as e:
            if e.errno != 17:  # FILE EXISTS
                raise e


# %%
class SmartProgressbar:
    def __init__(self, *args, **kwargs):
        self.pbar = tqdm.tqdm(*args, **kwargs)
        self.lastVal = 0

    def update(self, val):
        self.pbar.update(val - self.lastVal)
        self.lastVal = val

    def finish(self):
        self.pbar.close()


# %%
def progress_bar(text, items):
    pbar = SmartProgressbar(total=items, leave=True, desc=text, file=sys.stderr,
                            unit_scale=True, ascii=True, mininterval=15, maxinterval=30)
    return pbar


def setup_logging(level=logging.DEBUG):
    """Configure root logging with a thread-aware, timestamped format.

    Args:
        level: Logging level passed to ``logging.basicConfig``.
    """
    logging.basicConfig(level=level, format='[%(levelname)s] [%(threadName)s] [%(asctime)s] %(message)s')


class EmptyContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, value, traceback):
        pass


def bb_iou(bb_a, bb_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_int1 = max(bb_a[0], bb_b[0])
    y_int1 = max(bb_a[1], bb_b[1])
    x_int2 = min(bb_a[2], bb_b[2])
    y_int2 = min(bb_a[3], bb_b[3])

    # compute the area of intersection rectangle
    int_area = max(0, (x_int2 - x_int1)) * max(0, (y_int2 - y_int1))

    # compute area of both bbs
    bb_a_area = (bb_a[2] - bb_a[0]) * (bb_a[3] - bb_a[1])
    bb_b_area = (bb_b[2] - bb_b[0]) * (bb_b[3] - bb_b[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = int_area / (bb_a_area + bb_b_area - int_area)
    return iou


def homogeneous(p):
    p = np.asarray(p).flatten()
    assert len(p) == 2 or len(p) == 3
    if len(p) == 2:
        return np.concatenate((p, [1]))
    else:
        return p / p[2]


def point_line_distance(p, l):
    return abs(np.dot(l, homogeneous(p))) / np.linalg.norm(l[0:2])


def line_from_points(p1, p2):
    return np.cross(homogeneous(p1), homogeneous(p2))


# %%
def init_visible_GPU(skip_on_nodes=("pcsochor", "pcspanhel-gpu"),
                     random_sleep_max=180, verbose=True):
    if platform.node() in skip_on_nodes:
        return

    sge_task_id = os.getenv("SGE_TASK_ID")
    if sge_task_id is not None:
        sge_task_id = int(sge_task_id)
        logging.debug("Setting random seed for GPU initialization to %d" % (sge_task_id))
        random.seed(sge_task_id)

    sleep_time = random.uniform(0, random_sleep_max)
    logging.debug("Sleeping for %f seconds" % sleep_time)
    time.sleep(sleep_time)
    logging.debug("Sleeping done.")

    if verbose:
        os.system("nvidia-smi")
    free_gpu = subprocess.check_output(
        'nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"',
        shell=True)
    if len(free_gpu) == 0:
        logging.error('No free GPU available!')
        sys.exit(1)

    os.environ['CUDA_VISIBLE_DEVICES'] = free_gpu.decode().strip()
    logging.debug("CUDA device id: %s" % os.environ['CUDA_VISIBLE_DEVICES'])
    logging.debug("Running tiny tensorflow graph to get the GPU immediately")

    import tensorflow as tf  # late import
    with tf.Session() as sess:
        x = tf.constant(2)
        sess.run(x)
    logging.debug("Tiny tensorflow graph done.")


# %%
def _hierarchy_print(d, indent=0):
    for key, value in sorted(d.items(), key=lambda i: i[0]):
        if isinstance(value, dict):
            print('\t' * indent + "%s:" % (key))
            _hierarchy_print(value, indent + 1)
        else:
            print('\t' * indent + "%s: %s" % (key, value))


def hierarchy_print(d, name="", show_header=True, show_footer=True, total_padding=50):
    padding_left = (total_padding - len(name)) // 2
    padding_right = total_padding - len(name) - padding_left
    if show_header:
        print("=" * padding_left + name + "=" * padding_right)
    _hierarchy_print(d)
    if show_footer:
        print("=" * total_padding)
