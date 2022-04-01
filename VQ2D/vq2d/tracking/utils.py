import cv2
import numpy as np

from ..structures import BBox


def draw_bbox(image: np.ndarray, bbox: BBox) -> np.ndarray:
    x1 = bbox.x1
    x2 = bbox.x2
    y1 = bbox.y1
    y2 = bbox.y2
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image
