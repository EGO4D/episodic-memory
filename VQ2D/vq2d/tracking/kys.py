import importlib

import cv2
import numpy as np
from pytracking.features.net_wrappers import NetWithBackbone

from ..structures import BBox
from .utils import draw_bbox


class KYSTracker(object):
    def __init__(self, model_path):
        name, parameter_name = "kys", "default"
        # Load tracker parameters
        param_module = importlib.import_module(
            "pytracking.parameter.{}.{}".format(name, parameter_name)
        )
        params = param_module.parameters()
        params.tracker_name = name
        params.param_name = parameter_name
        params.net = NetWithBackbone(net_path=model_path, use_gpu=params.use_gpu)

        # Create tracker
        tracker_module = importlib.import_module("pytracking.tracker.{}".format(name))
        tracker_class = tracker_module.get_tracker_class()
        self.tracker = tracker_class(params)
        self.state = None
        self._lost_track = None

    def initialize_tracker(self, frame, box):
        init_state = [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1]
        self.tracker.initialize(frame, self._build_init_info(init_state))
        self.state = init_state
        self._lost_track = False

    def _build_init_info(self, box):
        return {
            "init_bbox": box,
            "init_object_ids": [
                1,
            ],
            "object_ids": [
                1,
            ],
            "sequence_object_ids": [
                1,
            ],
        }

    def update_state(self, frame):
        out = self.tracker.track(frame)
        self.state = [int(s) for s in out["target_bbox"]]
        self._lost_track = self.tracker.debug_info["flag" + self.tracker.id_str]

    @property
    def lost_track(self):
        return self._lost_track == "not_found"


class KYSRunner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracker = KYSTracker(cfg.tracker.kys_tracker.model_path)

    def __call__(self, init_state, init_frame, search_frames, *args, **kwargs):
        return run_kys_tracker(
            init_state, init_frame, search_frames, self.cfg, self.tracker
        )


def run_kys_tracker(init_state, init_frame, search_frames, cfg, kys_tracker):

    kys_cfg = cfg.tracker.kys_tracker

    start_fno = init_state.fno

    global img_height, img_width
    img_height, img_width = init_frame.shape[:2]

    if cfg.logging.visualize:
        start_frame_vis = np.copy(init_frame)
        draw_bbox(start_frame_vis, init_state)
        start_frame_vis = cv2.resize(start_frame_vis, None, fx=0.5, fy=0.5)

    # -- BACKWARD TRACKING
    # initialize the tracker
    kys_tracker.initialize_tracker(init_frame, init_state)

    backward_track = []
    backward_track_vis = []

    start_rt_pred = start_fno
    for i in range(start_fno - 1, -1, -1):
        image = search_frames[i]  # RGB
        kys_tracker.update_state(image)
        if kys_tracker.lost_track:
            break

        start_rt_pred = i
        x, y, w, h = kys_tracker.state
        bbox = BBox(i, x, y, x + w, y + h)
        backward_track.append(bbox)

        if kys_cfg.debug or cfg.logging.visualize:
            draw_bbox(image, bbox)

            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            backward_track_vis.append(image)

        if kys_cfg.debug:
            cv2.imshow("Image", image[..., ::-1])
            cv2.waitKey(300)

    # Add a few padding frames
    if cfg.logging.visualize:
        for i in range(start_rt_pred - 1, max(start_rt_pred - 10, 1), -1):
            image = search_frames[i]
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            backward_track_vis.append(image)

    # -- FORWARD TRACKING
    # initialize the tracker
    kys_tracker.initialize_tracker(init_frame, init_state)

    forward_track = []
    forward_track_vis = []
    end_rt_pred = start_fno
    for i in range(start_fno + 1, len(search_frames), 1):
        image = search_frames[i]
        kys_tracker.update_state(image)
        if kys_tracker.lost_track:
            break

        end_rt_pred = i
        x, y, w, h = kys_tracker.state
        bbox = BBox(i, x, y, x + w, y + h)
        forward_track.append(bbox)

        if kys_cfg.debug or cfg.logging.visualize:
            draw_bbox(image, bbox)

            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            forward_track_vis.append(image)

        if kys_cfg.debug:
            cv2.imshow("Image", image[..., ::-1])
            cv2.waitKey(300)

    # Add a few padding frames
    if cfg.logging.visualize:
        for i in range(end_rt_pred + 1, min(end_rt_pred + 10, len(search_frames))):
            image = search_frames[i]
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            forward_track_vis.append(image)

    response_track = backward_track[::-1] + [init_state] + forward_track
    response_track_vis = None
    if cfg.logging.visualize:
        response_track_vis = (
            backward_track_vis[::-1] + [start_frame_vis] + forward_track_vis
        )

    return response_track, response_track_vis
