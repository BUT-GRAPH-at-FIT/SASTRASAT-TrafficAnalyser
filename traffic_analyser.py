#!/usr/bin/env python3
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import logging
from queue import Queue, Full, Empty
import cv2
import numpy as np
import os
import json
import zmq
import sys
import h5py
import csv

from tensorflow.keras.layers import DepthwiseConv2D, ReLU
from tensorflow.keras.models import load_model, Model
from tensorflow.compat.v1.keras.backend import set_session

from libsj.nn.object_detector import ObjectDetectorThread
from libsj.tracking import TrackerThread, IoUTracker, KCFTracker
from libsj.plotting import cv_draw_text
from libsj.video_io import VideoReader, VideoWriter
from libsj.utils import setup_logging, load_cache, save_cache, EmptyContext, ensure_dir, save_np_cache, progress_bar#, point_line_distance, line_from_points,
from libsj.threading import BaseThread, QueuesMonitorThread, ProcessingThread, QueueDuplicatorThread
# from libsj.traffic_calib import lstsq_lines_intersection, TrafficCalibration, optimize_vp2_by_measurements

from omegaconf import DictConfig, OmegaConf
import hydra


STATUS_BAR_HEIGHT = 30
QUEUE_SIZE = 256
DETECTOR_SCALE_FACTOR = 0.25

VIAN_TOKEN = "VRASSEO_ISPANHEL_TEST"
VIAN_SERVER_URL = "https://vian-dev.fit.vutbr.cz/vian_sensingapi/"
VIAN_PROJECT_ID = "test"

def parse_args():
    parser = argparse.ArgumentParser(description="Tensorflow object detection in video",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument("--config", "-c", type=str, required=True, help="path to json file with config")
    parser.add_argument("--frame-step", type=int, default=1, help="frames step")
    parser.add_argument("--skip-frames", type=int, default=0, help="skip x first frames")
    parser.add_argument("--take-frames", type=int, default=None, help="take only x frames, default all")
    parser.add_argument("--detection-model", type=str, help="detector frozen graph path", required=True)
    parser.add_argument("--detection-gpu-mem", type=float, help="fraction of gpu memory to use for detection", default=0.20)
    parser.add_argument("--detection-threshold", type=float, default=0.5, help="threshold for detection")
    parser.add_argument("--classification-model", type=str, help="classifier keras model (.h5)", required=True)
    parser.add_argument("--extractor-model", type=str, help="extractor model (.h5) for re-id features", required=True)
    parser.add_argument("--color-classification-model", type=str, default="models/classifiers/colors_MobileNet", help="color classifier keras model (.h5)")
    parser.add_argument("--tracker", type=str, default="IoU", help="tracker type")
    parser.add_argument("--tracker-iou", type=float, default=0.4, help="threshold for matching of tracks and detections")
    parser.add_argument("--tracker-terminate-after", type=int, default=5, help="terminate tracks after specified number of frames without detection")
    parser.add_argument("--video-output", default=None, type=str, help="output video file; if None, the output is shown on screen")
    parser.add_argument("--output", "-o", type=str, required=True, help="data output dir")
    parser.add_argument("--calibration-output", type=str, default=None, help="data output dir")
    parser.add_argument("video", type=str, help="path to input video")
    args = parser.parse_args()
    return args


class ClassificationThread(ProcessingThread):
    """
    Thread in processing pipeline.
    Feature vectors are computed for detected vehicles
    """

    # def __init__(self, model_path, input_queue, output_queue, name="Classifier"):
    def __init__(self, model_path, color_model_path, extractor_model_path, input_queue, output_queue,
                 collect_vehicle_crops=False, name="Classifier"):
        super().__init__(input_queue, output_queue, name)
        self.model_path = model_path
        self.color_model_path = color_model_path
        self.extractor_model_path = extractor_model_path
        self.collect_vehicle_crops = collect_vehicle_crops
        # Init session for keras models
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.keras_session = tf.compat.v1.Session(config=config)
        set_session(self.keras_session)

    def init_thread(self):
        self.model = load_model(os.path.join(self.model_path, "final_model.h5"),
                                custom_objects={'relu6': ReLU(6.), 'DepthwiseConv2D': DepthwiseConv2D}, compile=False)
        self.color_model = load_model(os.path.join(self.color_model_path, "final_model.h5"),
                                      custom_objects={'relu6': ReLU(6.), 'DepthwiseConv2D': DepthwiseConv2D}, compile=False)
        self.extractor_model = load_model(os.path.join(self.extractor_model_path, "model.h5"),
                                      custom_objects={'relu6': ReLU(6.), 'DepthwiseConv2D': DepthwiseConv2D}, compile=False)

        self.vehicle_extractor = Model(self.extractor_model.input, self.extractor_model.layers[-1].output, name="Extractor")

        self.image_size = (self.model.input_shape[2], self.model.input_shape[1])  # width, height
        mapping = load_cache(os.path.join(self.model_path, "types_mapping.pkl"))
        assert len(mapping) == self.model.output_shape[-1]
        self.mapping = {v: k for k, v in mapping.items()}
        mapping = load_cache(os.path.join(self.color_model_path, "colors_mapping.pkl"))
        assert len(mapping) == self.color_model.output_shape[-1]
        self.color_mapping = {v: k for k, v in mapping.items()}
        assert self.model.input_shape == self.color_model.input_shape

    # def process(self, data):
    #     frame = data["frame"]
    #     # crops = []
    #     for track_id, track in data["tracks"].items():
    #
    #         if track["status"] in {"new", "detected"} and track["class"] == 1:
    #             bb = np.asarray(track["bb"])[-1, 0:4]
    #             bb = np.round(bb * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]))
    #             x1, y1, x2, y2 = bb.astype(np.int32)
    #             crop = frame[y1:y2, x1:x2, :]
    #             crop = cv2.resize(crop, self.image_size)
    #             # crops.append(crop)
    #
    #             # if len(crops) > 0:
    #             np_crops = np.asarray([crop])
    #             np_crops = (np_crops.astype(np.float32) - 116.0) / 128.0
    #             predictions = self.model.predict(np_crops, batch_size=16)
    #             color_predictions = self.color_model.predict(np_crops, batch_size=16)
    #             classifications = []
    #             colors = []
    #             for pred, color_pred in zip(predictions, color_predictions):
    #                 ind = np.argmax(pred)
    #                 classifications.append((self.mapping[ind], pred[ind]))
    #                 ind = np.argmax(color_pred)
    #                 colors.append((self.color_mapping[ind], color_pred[ind]))
    #             track["classification"] = classifications[0]
    #             track["color"] = colors[0]
    #
    #     return data

    def process(self, data):
        # TODO: Upravit tak, aby se vzali detekce ze všech tracků, provedlo se rozpoznání a pak se přiradilo na základě ID tracku ke správným trackům
        frame = data["frame"]
        vehicle_crops = []
        vehicle_track_ids = []
        peds_crops = []
        peds_track_ids = []
        for track_id, track in data["tracks"].items():

            if track["status"] in {"new", "detected"}:
                bb = np.asarray(track["bb"])[-1, 0:4]
                bb = np.round(bb * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]))
                x1, y1, x2, y2 = bb.astype(np.int32)
                crop = frame[y1:y2, x1:x2, :]
                crop = cv2.resize(crop, self.image_size)

                if track["class"] == 1: # class 1 - vehicle
                    vehicle_crops.append(crop)
                    vehicle_track_ids.append(track_id)
                elif track["class"] == 2:
                    peds_crops.append(crop)
                    peds_track_ids.append(track_id)


        if len(vehicle_crops) > 0:
            np_crops = np.asarray(vehicle_crops)
            np_crops = (np_crops.astype(np.float32) - 116.0) / 128.0
            predictions = self.model.predict(np_crops, batch_size=16)

            # Car classification
            color_predictions = self.color_model.predict(np_crops, batch_size=16)

            # extract feautres from vehicles
            feats = self.vehicle_extractor.predict(np_crops, batch_size=16)
            feats /= np.linalg.norm(feats, axis=1, keepdims=True)

            for idx, (track_id, pred, color_pred, feat) in enumerate(zip(vehicle_track_ids, predictions, color_predictions, feats)):
                ind = np.argmax(pred)
                classification = (self.mapping[ind], pred[ind])
                ind = np.argmax(color_pred)
                color = (self.color_mapping[ind], color_pred[ind])
                data["tracks"][track_id]["classification"] = classification
                data["tracks"][track_id]["color"] = color
                data["tracks"][track_id]["feature"] = feat

                if self.collect_vehicle_crops:
                    data["tracks"][track_id]["crop"] = vehicle_crops[idx]

        if len(peds_crops) > 0:
            np_crops = np.asarray(peds_crops)
            np_crops = (np_crops.astype(np.float32) - 116.0) / 128.0

            #TODO: extract feautres from vehicles
            feats = self.vehicle_extractor.predict(np_crops, batch_size=16)
            feats /= np.linalg.norm(feats, axis=1, keepdims=True)

            for track_id, feat in zip(peds_track_ids, feats):
                data["tracks"][track_id]["classification"] = None
                data["tracks"][track_id]["color"] = None
                data["tracks"][track_id]["feature"] = feat
        return data

    def finalize_thread(self):
        pass


class AnalyseThread(ProcessingThread):
    """
    Thread in processing pipeline.
    All traffic situation analysis is done in this thread.
    This includes: estimation of semaphore state, violation detection, and others.
    """

    def __init__(self, config, input_queue, output_queue, calibration_output, name="Analyse"):
        super().__init__(input_queue, output_queue, name)
        self.config = config
        self.semaphore_state = None
        self.semaphore_M = None
        self.semaphore_total = 0
        self.calib = config["calib"]
        self.calibration_output = calibration_output
        self.calibration_drawn = False
        self.heights = {}

    def get_semaphore_state(self, frame):
        # global semaphore_img
        semaphore_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.semaphore_M is None:
            pts1 = np.float32(self.config["semaphore"])
            pts2 = np.float32([[0, 0], [100, 0], [100, 300], [0, 300]])
            self.semaphore_M = cv2.getPerspectiveTransform(pts1, pts2)
            semaphore_img = cv2.warpPerspective(semaphore_img, self.semaphore_M, (100, 300))
        else:
            semaphore_img = cv2.warpPerspective(semaphore_img, self.semaphore_M, (100, 300))

        hsv = cv2.cvtColor(semaphore_img, cv2.COLOR_BGR2HSV)

        red = cv2.inRange(hsv[0:100, 0:100], np.array([0, 60, 30]), np.array([15, 100, 100]))
        orange = cv2.inRange(hsv[100:200, 0:100], np.array([20, 60, 30]), np.array([50, 100, 100]))
        green = cv2.inRange(hsv[200:300, 0:100], np.array([95, 60, 30]), np.array([145, 100, 100]))

        colors = []
        colors.append(cv2.countNonZero(red))
        colors.append(cv2.countNonZero(orange))
        colors.append(cv2.countNonZero(green))
        total = sum(colors)
        max_value = max(colors)

        state = colors.index(max_value)
        if (max_value > 75):
            self.semaphore_state = state
            self.semaphore_total = total

        if self.semaphore_state is None:
            self.semaphore_state = state
            self.semaphore_total = total

        state_text = ["red", "orange", "green"]

        return state_text[self.semaphore_state]

    def get_violation(self, track, semaphore, pedestrian_in_protected):
        if int(track["class"]) == 1:  # vehicle
            in_protected_area = cv2.pointPolygonTest(self.config["protected_area"], track["position"], False) > 0
            if in_protected_area and pedestrian_in_protected and track["is_moving"]:
                return "vehicle_protected_moving"

            if semaphore == "red" and in_protected_area:
                return "vehicle_protected_red"

        else:  # pedestrian
            if semaphore in ("green", "orange") and track["in_protected"]:
                return "pedestrian_crossing_green"
        return None

    def draw_calibration(self, frame):
        if not self.calibration_drawn and self.calibration_output is not None:
            fig, axs = plt.subplots(nrows=2, figsize=(18, 20))
            axs[0].imshow(frame)
            for key in ["semaphore", "roi", "protected_area", "crossing"]:
                axs[0].plot(config[key][list(range(config[key].shape[0])) + [0], 0],
                            config[key][list(range(config[key].shape[0])) + [0], 1],
                            lw=5, ls="--", label=key)
            axs[0].legend()
            self.calib.plot_grid(axs[1], vp1_dir_mults=(-10, 15), vp2_dir_mults=(-10, 10))
            axs[1].imshow(frame)
            plt.tight_layout()
            plt.savefig(self.calibration_output)
            logging.info("Calibration image saved to: %s" % self.calibration_output)
            self.calibration_drawn = True

    def process(self, data):
        frame = data["frame"]
        # self.draw_calibration(frame)
        # semaphore = self.get_semaphore_state(frame)
        semaphore = "None"
        stats = {1: 0, 2: 0}
        violations_num = 0
        pedestrian_in_protected = False
        # fill in_roi, in_protected and compute stats + pedestrian_in_protected
        for track_id, track in data["tracks"].items():
            track["in_roi"] = False
            track["in_protected"] = False
            track["is_moving"] = False
            track["movement"] = None
            if track["status"] in {"new", "detected"}:
                bb = np.asarray(track["bb"])[-1, 0:4]
                if int(track["class"]) == 1:  # center for vehicles
                    position = (int((bb[0] + bb[2]) / 2 * frame.shape[1]), int((bb[1] + bb[3]) / 2 * frame.shape[0]))
                    # vehicle movement
                    if len(np.asarray(track["bb"])) > 2:
                        # 2 points back
                        prev_bb = np.asarray(track["bb"])[-3, 0:4]
                        prev_position = (int((prev_bb[0] + prev_bb[2]) / 2 * frame.shape[1]),
                                         int((prev_bb[1] + prev_bb[3]) / 2 * frame.shape[0]))
                        # store movement (vx, vy)
                        track["movement"] = (np.asarray(position) - prev_position)
                        # movement speed (in pixels)
                        track["speed"] = np.linalg.norm(track["movement"])
                        # evaluate movement
                        track["is_moving"] = (np.linalg.norm(track["movement"]) > 10)
                else:  # bottom center for pedestrians
                    position = (int((bb[0] + bb[2]) / 2 * frame.shape[1]), int(bb[3] * frame.shape[0]))
                    top = (int((bb[0] + bb[2]) / 2 * frame.shape[1]), int(bb[1] * frame.shape[0]))
                    # current_height = self.calib.height(top, position)
                    # if track_id not in self.heights:
                    #     self.heights[track_id] = []
                    # self.heights[track_id].append(current_height)
                    # track["height"] = np.median(self.heights[track_id])
                    track["height"] = 0
                track["position"] = position
                # track["in_roi"] = cv2.pointPolygonTest(self.config["roi"], position, False) > 0
                # track["in_protected"] = cv2.pointPolygonTest(self.config["protected_area"], position, False) > 0
                # if track["in_protected"] and int(track["class"]) == 2:
                #     pedestrian_in_protected = True
                # if track["in_roi"]:
                #     stats[int(track["class"])] += 1
            elif track["status"] == "terminated":
                if track_id in self.heights:
                    del self.heights[track_id]
        # fill violations
        for track_id, track in data["tracks"].items():
            track["violation"] = None
            if track["status"] in {"new", "detected"}:
                # track["violation"] = self.get_violation(track, semaphore, pedestrian_in_protected)
                pass
            if track["violation"] is not None:
                violations_num += 1

        data["stats"] = {"vehicles": stats[1], "pedestrians": stats[2],
                         "violations": violations_num, "semaphore": semaphore,
                          "protected_area": "None", #self.config["protected_area"],
                          "pedestrian_in_protected": 0 # pedestrian_in_protected
                         }

        if (data["frame_id"] % 5000) == 0 and data["frame_id"] > 0:
            logging.debug("Processed %d frames" % (data["frame_id"]))
        return data



class DrawThread(ProcessingThread):
    """
    Thread in processing pipeline.
    Draw current situation to the frame
    """

    def __init__(self, input_queue, output_queue, name="Draw"):
        super().__init__(input_queue, output_queue, name)
        self.colors = [None, (0, 255, 0), (255, 0, 0)]
        self.violation_color = {"vehicle_protected_moving": (0, 0, 255),
                                "vehicle_protected_red": (0, 165, 255),
                                "pedestrian_crossing_green": (0, 255, 255)}
        self.vulnarable_user_color = (239, 255, 0)

    def _get_color(self, track_data, classification_data):
        # if track_data["violation"] is not None:
        #     return self.violation_color[track_data["violation"]]
        # if int(track_data["class"]) == 2 and track_data["height"] < 1.5:
        #     return self.vulnarable_user_color
        return self.colors[int(track_data["class"])]

    def process(self, data):
        frame_id = data["frame_id"]
        frame = np.ascontiguousarray(data["frame"][:, :, ::-1])
        # cv2.polylines(frame, data["stats"]["protected_area"].reshape(1, -1, 2).astype(np.int32), True, (255, 255, 255),
        #               4, cv2.LINE_AA)
        for track_id, track in data["tracks"].items():
            # if track["in_roi"] and track["status"] in {"new", "detected"}:
            if track["status"] in {"new", "detected"}:
                color = self._get_color(track, None)
                boxes = np.round(np.asarray(track["bb"])[:, 0:4] * np.array(
                    [[frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]]))
                x1, y1, x2, y2 = boxes[-1].astype(np.int32)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                cv2.circle(frame, tuple(track["position"]), 3, color, 3, cv2.LINE_AA)
                if track["is_moving"] and int(track["class"]) == 1: #Class 1 - Car, Class 2 - person
                    p = np.asarray(track["position"])
                    v = np.asarray(track["movement"])
                    v = v / np.linalg.norm(v) * 40
                    cv2.arrowedLine(frame, tuple(p), tuple(np.round(p + v).astype(int)), color, 2, cv2.LINE_AA)



                if "height" in track: # persons
                    if np.mean(color) > 100:
                        text_color = (0, 0, 0)
                    else:
                        text_color = (255, 255, 255)
                    cv_draw_text(frame, "ID:%04d,H:%.1fm" % (track_id,track["height"]), (x1 + 1, y1),
                                 background_color=color, text_color=text_color, padding=3,
                                 font_scale=0.4, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)
                else: # cars
                    # classifications
                    vehicle_type, vehicle_type_p = track["classification"]
                    vehicle_color, vehicle_color_p = track["color"]

                    text_color = (0,0,0)
                    cv_draw_text(frame, "ID:%04d, Type: %s (%.02f), Color: %s (%.02f)" %
                                 (track_id, vehicle_type, vehicle_type_p, vehicle_color, vehicle_color_p),
                                 (x1 + 1, y1),
                                 background_color=color, text_color=text_color, padding=3,
                                 font_scale=0.4, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)



        status_text = "frame: %d  vehicles: %d  pedestrians: %d  violations: %d  semaphore: %s  crossing: %s" % (
            frame_id, data["stats"]["vehicles"], data["stats"]["pedestrians"],
            data["stats"]["violations"], data["stats"]["semaphore"], str(data["stats"]["pedestrian_in_protected"]))
        frame = np.concatenate(
            (np.ones((STATUS_BAR_HEIGHT, frame.shape[1], frame.shape[2]), dtype=np.uint8) * 255, frame))
        cv_draw_text(frame, status_text, (5, 20), background_color=(255, 255, 255), padding=10,
                     font_scale=0.50, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)
        return frame

class ShowThread(ProcessingThread):
    """
    Thread in processing pipeline.
    Shows the situation on screen.
    """

    def __init__(self, input_queue, window_name="Traffic Analysis", name="Show"):
        super().__init__(input_queue, None, name)
        self.window_name = window_name

    def init_thread(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def process(self, data):
        #global semaphore_img
        cv2.imshow(self.window_name, data)
        cv2.waitKey(1)

    def finalize_thread(self):
        cv2.destroyWindow(self.window_name)


class DataOutputThread(ProcessingThread):
    """
    Thread in processing pipeline.
    Saves images of vehicles and their feature vectors.
    """

    def __init__(self, input_queue, output_dir, zmq_ip_addr="tcp://localhost:5555",
                 save_vehicle_crops=False, name="DataOutput"):
        super().__init__(input_queue, None, name)
        self.output_dir = output_dir
        self.zmq_ip_addr = zmq_ip_addr
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_socket.connect(self.zmq_ip_addr)
        self.save_vehicle_crops = save_vehicle_crops

        self.data_to_store = []

    def init_thread(self):
        ensure_dir(self.output_dir)
        self.crops = {}
        self.features = {}

    def process(self, data):
        if self.save_vehicle_crops:
            ensure_dir(os.path.join(self.output_dir, "vehicle_crops"))

        meta_file_path = os.path.join(self.output_dir, 'track_meta.csv')

        if not os.path.exists(meta_file_path):
            with open(meta_file_path, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["track_id", "frame_id", "position", "confidence", "crop_path"])

        track_feats = {}

        for track_id, track in data["tracks"].items():
            if track["status"] in {"new", "detected"}:

                if not "height" in track: # cars
                    meta = {
                        "track_id": int(track_id),
                        "frame_id": track["frame_id"],
                        "position": track["position"],
                        "confidence": float(track["score"]),
                        "crop_path": f"vehicle_{track_id}.jpg" if self.save_vehicle_crops else None
                    }
                    track_feats[meta["track_id"]] = [
                        float(x) for x in (track["feature"] if "feature" in track.keys() else [])
                    ]

                    if self.save_vehicle_crops:
                        vehicle_crop_np = track["crop"] if "crop" in track.keys() else None
                        if vehicle_crop_np is not None:
                            vehicle_crop = cv2.cvtColor(vehicle_crop_np, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(self.output_dir, "vehicle_crops", f"vehicle_{track_id}.jpg"), vehicle_crop)

                    with open(meta_file_path, 'a') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(list(meta.values()))

        with h5py.File(os.path.join(self.output_dir, 'features.h5'), 'a') as f:
            if "track_ids" not in f:
                f.create_dataset("track_ids", data=list(track_feats.keys()), maxshape=(None,), chunks=True)
                f.create_dataset("features", data=list(track_feats.values()), maxshape=(None, 128), chunks=True)
            else:
                n = len(f['track_ids'])
                new_track_ids = list(track_feats.keys())
                new_features = list(track_feats.values())

                f["features"].resize((n + len(new_features), 128))
                f["features"][n:] = new_features
                f["track_ids"].resize((n + len(new_track_ids),))
                f["track_ids"][n:] = new_track_ids


def main(cfg: DictConfig) -> None:
    setup_logging(logging.DEBUG)
    # args = parse_args()
    
    # Intialize Hydra config
    # initialize(config_path="config")  # Inicializujte Hydra s cestou k adresáři konfigurace
    # cfg = compose(config_name="config")  # Načtěte konfigurační soubor config.yaml
    # initialize_config_module(config_module="config")
    setup_logging(logging.DEBUG)
    # cfg = compose(config_name="config", overrides=["video=test.mp4", "output=test_out"])

    print(OmegaConf.to_yaml(cfg))  # Vypište konfiguraci

    # experiment path: year_month/day/hour/minute
    camera_name = cfg.video.source.split("/")[-1].split(".")[0]
    experiment_path = datetime.now().strftime(f"{cfg.output.data_dir}/{camera_name}/%Y_%m/%d/%H/%M/")
    ensure_dir(experiment_path)

    all_queues = []
    all_threads = []
    try:
        # READER
        reader = VideoReader(cfg.video.source,
                             frame_step=cfg.video.frame_step,
                             skip_frames=cfg.video.skip_frames,
                             take_frames=cfg.video.take_frames,
                             queue_max_size=QUEUE_SIZE,
                             av_options={"rtsp_transport":"tcp","buffer_size":"2048","prefer_tcp":"1"})
        all_threads.append(reader)
        all_queues.append(("detector_in", reader.queue))

        # GET CONFIG
        # config = get_config(args.config, reader.frame_shape[0:2])
        config = {"calib": None}

        # DETECTOR
        detections_output_queue = Queue(QUEUE_SIZE)
        detector = ObjectDetectorThread(cfg.detection.model, reader.queue, detections_output_queue,
                                        gpu_mem=cfg.detection.gpu_mem,
                                        allow_growth=True,  #TODO: nastavit z parametrů??
                                        detector_scale_factor=DETECTOR_SCALE_FACTOR,
                                        threshold=cfg.detection.threshold)

        all_threads.append(detector)
        all_queues.append(("tracker_in", detections_output_queue))

        # TRACKER
        tracker_output_queue = Queue(QUEUE_SIZE)
        if cfg.tracker.type == "IoU":
            tracker_impl = IoUTracker(cfg.tracker.iou, cfg.tracker.terminate_after)
        elif cfg.tracker.type == "KCF":
            tracker_impl = KCFTracker(cfg.tracker.iou, cfg.tracker.terminate_after)
        else:
            raise ValueError("Unsupported tracker: '%s'" % cfg.tracker.type)
        tracker = TrackerThread(tracker_impl, detections_output_queue, tracker_output_queue)
        all_threads.append(tracker)
        all_queues.append(("classifier_in", tracker_output_queue))

        # CLASSIFIER
        classifier_output_queue = Queue(QUEUE_SIZE)
        classifier = ClassificationThread(cfg.classification.mmr_model, cfg.classification.color_model, cfg.extractor.model,
                                          tracker_output_queue, classifier_output_queue, collect_vehicle_crops=cfg.output.save_crops)
        all_threads.append(classifier)
        all_queues.append(("analyser_in", classifier_output_queue))
        # all_queues.append(("splitter_in", classifier_output_queue))

        # ANALYSER
        analyser_output_queue = Queue(QUEUE_SIZE)
        analyser = AnalyseThread(config, classifier_output_queue, analyser_output_queue, cfg.output.data_dir)
        # analyser = AnalyseThread(config, tracker_output_queue, analyser_output_queue, args.calibration_output)
        all_threads.append(analyser)
        all_queues.append(("splitter_in", analyser_output_queue))

        # SPLIT DATA
        draw_queue_in = Queue(QUEUE_SIZE)
        data_output_queue_in = Queue(QUEUE_SIZE)
        duplicator = QueueDuplicatorThread(analyser_output_queue, [draw_queue_in, data_output_queue_in])
        all_threads.append(duplicator)
        all_queues.append(("draw_in", draw_queue_in))
        all_queues.append(("data_output_in", data_output_queue_in))

        # DRAW
        draw_output_queue = Queue(QUEUE_SIZE)
        drawer = DrawThread(draw_queue_in, draw_output_queue)
        all_threads.append(drawer)
        all_queues.append(("video_output_in", draw_output_queue))

        # QUEUE MONITOR
        monitor = QueuesMonitorThread(all_queues, report_period=60)
        all_threads.append(monitor)

        # VIDEO OUTPUT
        if cfg.output.video_output is None:
            video_output = ShowThread(draw_output_queue)
        else:
            height, width = reader.frame_shape[0:2]
            video_path = os.path.join(experiment_path, cfg.output.video_output.file)
            video_output = VideoWriter(video_path, width, height + STATUS_BAR_HEIGHT, codec="mpeg4",
                                       fps=15, input_queue=draw_output_queue, swap_channels=True)
        all_threads.append(video_output)

        # DATA OUTPUT
        data_output = DataOutputThread(data_output_queue_in, experiment_path,
                                       save_vehicle_crops=cfg.output.save_crops)
        all_threads.append(data_output)

        # START THREADS
        for t in all_threads:
            t.start()

        # JOIN THE DATA OUTPUT
        data_output.join()
    finally:
        for t in all_threads:
            t.exit()


@hydra.main(config_path="config", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)

if __name__ == "__main__":
    hydra_main()