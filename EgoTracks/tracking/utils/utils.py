import logging
from typing import List

import av
import cv2
from tracking.utils.env import pathmgr


def opencv_loader(path):
    """Read image using opencv's imread function and returns it in rgb format"""
    try:
        path = pathmgr.get_local_path(path)
        im = cv2.imread(path, cv2.IMREAD_COLOR)

        # convert to rgb and return
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.info('ERROR: Could not read image "{}"'.format(path))
        logging.info(e)
        return None


def visualize_bbox(frames: List, bboxes: List, texts: List = None) -> List:
    imgs = []
    for i in range(len(frames)):
        img = frames[i]
        box = bboxes[i]
        text = str(texts[i]) if texts is not None else ""
        x, y, w, h = box
        color = (255, 255, 255)

        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 5)
        img = cv2.putText(
            img,
            text,
            (int(x), int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,  # FontScale.
            (0, 255, 0),  # Color.
            2,  # Thickness.
            cv2.LINE_AA,
        )
        imgs.append(img)

    return imgs


def extract_frames(frame_numbers: List, container: av.container.Container) -> List:
    frames = []
    frame_numbers = set(frame_numbers)
    for frame in container.decode(video=0):
        if frame.index in frame_numbers:
            frames.append(frame.to_ndarray(format="bgr24"))

    return frames


def extract_frames_by_range(
    start: int, end: int, container: av.container.Container
) -> List:
    frames = []
    for frame in container.decode(video=0):
        if frame.index >= start and frame.index <= end:
            frames.append(frame.to_ndarray(format="bgr24"))

    return frames


def pad_bboxes(frame_bbox_dict, frame_numbers):
    bboxes = []
    for n in frame_numbers:
        if n in frame_bbox_dict:
            bboxes.append(frame_bbox_dict[n])
        else:
            bboxes.append([0, 0, 0, 0])

    return bboxes
