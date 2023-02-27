import os
import sys

import cv2
import numpy as np
import torch
from einops import asnumpy
from scipy.stats import norm

from ..baselines.utils import resize_if_needed
from ..structures import BBox
from .pfilter import (
    ParticleFilter,
    independent_sample,
)
from .utils import draw_bbox


def observe(x, image, template_size):
    """
    create observation hypothesis given a particle and the current observed frame. each row in x contains one
    particle info.
    One row of x = [r,c,sr,sc].
    resize the current hyposthesis to template_size for comparision afterwards
    """

    image_max = np.array([image.shape[0], image.shape[1]])
    template_size = np.array(
        (max(1, template_size[0]), max(1, template_size[1])), dtype=int
    )  # (h, w)

    y = np.zeros((len(x), template_size[0], template_size[1], 3), dtype=float)

    for i, particle in enumerate(x):
        r, c, sr, sc = particle

        scale = np.array([sr, sc])
        hypo_dim = np.multiply(scale, template_size).round().astype(int)

        r = r.round().astype(int)
        c = c.round().astype(int)

        center = np.array([r, c])

        hypo_min = center - hypo_dim // 2
        hypo_min = hypo_min.round().astype(int)
        hypo_max = hypo_min + hypo_dim

        if np.all(hypo_min >= 0) and np.all(hypo_max < image_max):
            image_cutout = image[
                hypo_min[0] : hypo_max[0], hypo_min[1] : hypo_max[1], :
            ]

            if np.any(np.array(image_cutout.shape) <= 0):
                hypo = (
                    np.zeros((template_size[0], template_size[1], 3), dtype=float) - 1
                )
            else:
                hypo = cv2.resize(
                    image_cutout, (template_size[1], template_size[0])
                ).astype(float)
        else:
            hypo = np.zeros((template_size[0], template_size[1], 3), dtype=float) - 1
        y[i, :, :, :] = hypo
    return y


def deep_metric(x, y):
    """
    x - (N, F) numpy array
    y - (1, F) numpy array
    """
    with torch.no_grad():
        x_f = torch.nn.functional.normalize(x, dim=1)
        y_f = torch.nn.functional.normalize(y, dim=1)
        similarity = asnumpy(torch.nn.functional.cosine_similarity(x_f, y_f, dim=1))
    return similarity


def metric(x, y, sigma=1):
    h, w = x.shape[1:3]
    mean_squared_error = ((x.astype(float) / 255.0 - y.astype(float) / 255.0) ** 2).sum(
        axis=1
    )
    mean_squared_error = mean_squared_error.sum(axis=1)
    mean_squared_error = mean_squared_error.sum(axis=1)
    mean_squared_error = mean_squared_error / (h * w * 3)
    similarity = np.exp(-mean_squared_error / (2 * sigma ** 2))
    return similarity


class PFRunner(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def __call__(
        self,
        init_state,
        init_frame,
        video_reader,
        oshape,
        end_frame,
        net,
        *args,
        **kwargs
    ):
        return run_pfilter(
            init_state,
            init_frame,
            video_reader,
            oshape,
            end_frame,
            self.cfg,
            net,
            self.device,
        )


def run_pfilter(
    init_state, init_frame, video_reader, oshape, end_frame, cfg, net, device
):
    """
    init_state: initial state of the tracked obj in the frame (gathered from the
                detection stage)
                [x,y,sx, sy] (relative to oshape images)
    init_frame: frame corresponding to init_state (may be smaller than oshape)
    video_reader: reader which yields frames from the video
    oshape: (original width, original height) of Ego4D dataset frames
    end_frame: last frame in the search window + 1

    **Important note:**
        The frames from video_reader may be smaller than oshape.  These are
        upsampled to oshape, and final predictions are relative to oshape.
    """
    pf_cfg = cfg.tracker.pfilter

    owidth, oheight = oshape
    oshapeby2 = (owidth // 2, oheight // 2)
    init_frame = resize_if_needed(init_frame, oshape)

    global img_height, img_width
    img_height, img_width = init_frame.shape[:2]

    # set init states
    x1_init = init_state.x1
    x2_init = init_state.x2
    y1_init = init_state.y1
    y2_init = init_state.y2

    template = init_frame[y1_init:y2_init, x1_init:x2_init, :]  # RGB

    state_variables = ["r", "c", "sr", "sc"]

    r = int(np.round((y1_init + y2_init) / 2.0))
    c = int(np.round((x1_init + x2_init) / 2.0))
    sr = 1.0
    sc = 1.0

    # prior sampling function for each variable
    # (assumes x and y are coordinates in the range 0-img_size)
    # change prior function to take into account detection centers
    prior_fn = independent_sample(
        [
            norm(loc=r, scale=img_height * 0.07).rvs,
            norm(loc=c, scale=img_width * 0.07).rvs,
            norm(loc=sr, scale=0.1).rvs,
            norm(loc=sc, scale=0.1).rvs,
        ]
    )

    # start frame
    start_fno = init_state.fno

    if pf_cfg.use_deep_similarity:
        weight_fn = lambda x, y: deep_metric(x, y)
    else:
        weight_fn = lambda x, y: metric(x, y, sigma=pf_cfg.metric_sigma)

    if cfg.logging.visualize:
        start_frame_vis = np.copy(init_frame)
        draw_bbox(start_frame_vis, init_state)
        start_frame_vis = resize_if_needed(start_frame_vis, oshapeby2)

    # -- BACKWARD TRACKING
    # create the filter
    pf = ParticleFilter(
        prior_fn=prior_fn,
        init_template=template,
        observe_fn=observe,
        n_particles=pf_cfg.n_particles,
        dynamics_fn=None,
        noise_fn=None,
        weight_fn=weight_fn,
        use_deep_similarity=pf_cfg.use_deep_similarity,
        similarity_net=net,
        device=device,
        resample_proportion=pf_cfg.resample_proportion,
        column_names=state_variables,
    )

    if pf_cfg.debug:
        cv2.imshow("Template", template[..., ::-1])

    backward_track = []
    backward_track_vis = []

    start_rt_pred = start_fno
    for i in range(start_fno - 1, -1, -1):
        image = resize_if_needed(video_reader[i], oshape)  # RGB
        pf.update(image)

        state = pf.map_state
        score = pf.map_similarity
        template_size = np.array(pf.template.shape[:2])

        r, c, sr, sc = state
        scale = np.array([sr, sc])
        state_dim = np.multiply(scale, template_size).round().astype(int)
        r = r.round().astype(int)
        c = c.round().astype(int)
        center = np.array([r, c])
        state_min = center - state_dim // 2
        state_min = state_min.round().astype(int)
        state_max = state_min + state_dim

        bbox = BBox(i, state_min[1], state_min[0], state_max[1], state_max[0])
        backward_track.append(bbox)

        if pf_cfg.debug or cfg.logging.visualize:
            image = image.copy()
            draw_bbox(image, bbox)

            image = cv2.drawMarker(
                image,
                (c, r),
                (255, 0, 0),
                markerType=cv2.MARKER_STAR,
                markerSize=10,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            # pf.viz_particles(image)
            image = resize_if_needed(image, oshapeby2)
            backward_track_vis.append(image)

        if pf_cfg.debug:
            cv2.imshow("Image", image[..., ::-1])
            cv2.waitKey(300)

        if score < pf_cfg.sim_thresh:
            start_rt_pred = i
            break

    # Add a few padding frames
    if cfg.logging.visualize:
        for i in range(start_rt_pred - 1, max(start_rt_pred - 10, 1), -1):
            image = resize_if_needed(video_reader[i], oshapeby2)
            backward_track_vis.append(image)

    # -- FORWARD TRACKING
    # create the filter
    pf = ParticleFilter(
        prior_fn=prior_fn,
        init_template=template,
        observe_fn=observe,
        n_particles=pf_cfg.n_particles,
        dynamics_fn=None,
        noise_fn=None,
        weight_fn=weight_fn,
        use_deep_similarity=pf_cfg.use_deep_similarity,
        similarity_net=net,
        device=device,
        resample_proportion=pf_cfg.resample_proportion,
        column_names=state_variables,
    )

    end_rt_pred = start_fno
    forward_track = []
    forward_track_vis = []

    for i in range(start_fno + 1, end_frame):
        image = resize_if_needed(video_reader[i], oshape)
        pf.update(image)

        state = pf.map_state
        score = pf.map_similarity
        template_size = np.array(pf.template.shape[:2])

        r, c, sr, sc = state
        scale = np.array([sr, sc])
        state_dim = np.multiply(scale, template_size).round().astype(int)
        r = r.round().astype(int)
        c = c.round().astype(int)
        center = np.array([r, c])
        state_min = center - state_dim // 2
        state_min = state_min.round().astype(int)
        state_max = state_min + state_dim

        bbox = BBox(i, state_min[1], state_min[0], state_max[1], state_max[0])
        forward_track.append(bbox)

        if pf_cfg.debug or cfg.logging.visualize:
            image = image.copy()
            draw_bbox(image, bbox)

            image = cv2.drawMarker(
                image,
                (c, r),
                (255, 0, 0),
                markerType=cv2.MARKER_STAR,
                markerSize=10,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            # pf.viz_particles(image)
            image = resize_if_needed(image, oshapeby2)
            forward_track_vis.append(image)

        if pf_cfg.debug:
            cv2.imshow("Image", image[..., ::-1])
            cv2.waitKey(300)

        if score < pf_cfg.sim_thresh:
            end_rt_pred = i
            break

    # Add a few padding frames
    if cfg.logging.visualize:
        for i in range(end_rt_pred + 1, min(end_rt_pred + 10, end_frame)):
            image = resize_if_needed(video_reader[i], oshapeby2)
            forward_track_vis.append(image)

    response_track = backward_track[::-1] + [init_state] + forward_track
    response_track_vis = None
    if cfg.logging.visualize:
        response_track_vis = (
            backward_track_vis[::-1] + [start_frame_vis] + forward_track_vis
        )

    return response_track, response_track_vis
