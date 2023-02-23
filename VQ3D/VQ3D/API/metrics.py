import os
import sys
import numpy as np

sys.path.append('../annotation_API/API/')
from bounding_box import BoundingBox

class distL2():
    def compute(self, v1:np.ndarray, v2:np.ndarray) -> float:
        d = np.linalg.norm(v1-v2)
        return d

class angularError():
    def compute(self, v1:np.ndarray, v2:np.ndarray) -> float:
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return angle


Rz_90 = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2), 0, 0],
                  [np.sin(-np.pi/2),  np.cos(-np.pi/2), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                 ])

class accuracy():
    def compute(self, t:np.ndarray, box1:BoundingBox, box2:BoundingBox) -> bool:

        c = (box1.center + box2.center) / 2.0
        
        c = np.append(c, 1.)
        c = np.matmul(Rz_90, c)
        c = c[:3] / c[3]

        d = np.linalg.norm(c-t)
        d_gt = np.linalg.norm(box1.center - box2.center)

        diag1 = np.sqrt(np.sum(box1.sizes**2))
        diag2 = np.sqrt(np.sum(box2.sizes**2))

        m = np.mean([diag1, diag2])
        delta = np.exp(-m)

        return d < 6*(d_gt + delta)
