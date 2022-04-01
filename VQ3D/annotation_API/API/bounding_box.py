import os
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

class BoundingBox():
    def __init__(self, data: Any = None, scale=1.0):
        self.scale = scale
        if data:
            self.load(data)

    def load(self, data: Any ) -> None:
        if type(data) is dict:
            position = data['position']
            rotation = data['rotation']
            dimension = data['dimension']
        elif type(data) is str:
            f = json.load(open(data, 'r'))

            position = f['frames'][0]['items'][0]['position']
            rotation = f['frames'][0]['items'][0]['rotation']
            dimension = f['frames'][0]['items'][0]['dimension']
        else:
            raise NotImplementedError

        self.center = np.array([position['x'],
                                position['y'],
                                position['z']]) * self.scale

        self.sizes = np.array([dimension['x'],
                               dimension['y'],
                               dimension['z']]) * self.scale

        self.rotation = np.array([rotation['x'],
                                  rotation['y'],
                                  rotation['z'],
                                 ])

    def volume(self) -> float:
        return np.prod(self.sizes)

    def get_transformation_matrix(self) -> np.ndarray:
        rotation = self.rotation
        position = self.center

        Rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(rotation[0]), -np.sin(rotation[0]), 0],
                       [0, np.sin(rotation[0]),  np.cos(rotation[0]), 0],
                       [0, 0, 0, 1],
                      ])

        Ry = np.array([[ np.cos(rotation[1]), 0, np.sin(rotation[1]), 0],
                       [0, 1, 0, 0],
                       [-np.sin(rotation[1]), 0, np.cos(rotation[1]), 0],
                       [0, 0, 0, 1],
                      ])

        Rz = np.array([[np.cos(rotation[2]), -np.sin(rotation[2]), 0, 0],
                       [np.sin(rotation[2]),  np.cos(rotation[2]), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                      ])

        Tr = np.array([[1, 0, 0, position[0]],
                       [0, 1, 0, position[1]],
                       [0, 0, 1, position[2]],
                       [0, 0, 0, 1],
                      ])


        Rz_90 = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2), 0, 0],
                          [np.sin(-np.pi/2),  np.cos(-np.pi/2), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1],
                         ])


        return np.matmul(Rz_90, np.matmul(Tr, np.matmul(Ry, np.matmul(Rz, Rx))))

    def build_box(self) -> List[np.array]:

        sizes = self.sizes

        T = self.get_transformation_matrix()

        v0 = np.zeros(4); v0[3] = 1; v0[:3] -= sizes/2
        v1 = np.zeros(4); v1[3] = 1; v1[:2] -= sizes[:2]/2; v1[2]+=sizes[2]/2
        v2 = np.zeros(4); v2[3] = 1; v2[0]  -= sizes[0]/2;  v2[1]+=sizes[1]/2; v2[2] -= sizes[2]/2
        v3 = np.zeros(4); v3[3] = 1; v3[0]  -= sizes[0]/2;  v3[1]+=sizes[1]/2; v3[2] += sizes[2]/2
        v4 = np.zeros(4); v4[3] = 1; v4[:2] += sizes[:2]/2; v4[2]+=sizes[2]/2
        v5 = np.zeros(4); v5[3] = 1; v5[0]  += sizes[0]/2;  v5[1]+=sizes[1]/2; v5[2] -= sizes[2]/2
        v6 = np.zeros(4); v6[3] = 1; v6[0]  += sizes[0]/2;  v6[1]-=sizes[1]/2; v6[2] += sizes[2]/2
        v7 = np.zeros(4); v7[3] = 1; v7[0]  += sizes[0]/2;  v7[1]-=sizes[1]/2; v7[2] -= sizes[2]/2

        v0 = np.matmul(T, v0) # - bottom back left
        v1 = np.matmul(T, v1) # - top back left
        v2 = np.matmul(T, v2) # - bottom front left
        v3 = np.matmul(T, v3) # - top front left
        v4 = np.matmul(T, v4) # - top front right
        v5 = np.matmul(T, v5) # - bottom front right
        v6 = np.matmul(T, v6) # - top back right
        v7 = np.matmul(T, v7) # - bottom back right

        return [v0,v1,v2,v3,v4,v5,v6,v7]

    def save_off(self, filename: str) -> None:
        v0,v1,v2,v3,v4,v5,v6,v7 = self.build_box()

        vertices = []
        vertices.append(str(v0[0])+' '+str(v0[1])+' '+str(v0[2]))
        vertices.append(str(v1[0])+' '+str(v1[1])+' '+str(v1[2]))
        vertices.append(str(v2[0])+' '+str(v2[1])+' '+str(v2[2]))
        vertices.append(str(v3[0])+' '+str(v3[1])+' '+str(v3[2]))
        vertices.append(str(v4[0])+' '+str(v4[1])+' '+str(v4[2]))
        vertices.append(str(v5[0])+' '+str(v5[1])+' '+str(v5[2]))
        vertices.append(str(v6[0])+' '+str(v6[1])+' '+str(v6[2]))
        vertices.append(str(v7[0])+' '+str(v7[1])+' '+str(v7[2]))

        faces = []
        faces.append('4 '+ str(0)+ ' '+ str(1)+ ' '+ str(3)+ ' '+ str(2))
        faces.append('4 '+ str(0)+ ' '+ str(1)+ ' '+ str(6)+ ' '+ str(7))
        faces.append('4 '+ str(6)+ ' '+ str(7)+ ' '+ str(5)+ ' '+ str(4))
        faces.append('4 '+ str(5)+ ' '+ str(4)+ ' '+ str(3)+ ' '+ str(2))
        faces.append('4 '+ str(1)+ ' '+ str(3)+ ' '+ str(4)+ ' '+ str(6))
        faces.append('4 '+ str(0)+ ' '+ str(2)+ ' '+ str(5)+ ' '+ str(7))


        file = open(filename, 'w')
        file.write('OFF\n')
        file.write('{} {} 0\n'.format(str(len(vertices)),
                                      str(len(faces))))

        color = [255, 0, 0]

        for v in vertices:
            file.write(v+'\n')
        for f in faces[:-1]:
            line = f + ' ' + str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]) + ' 125\n'
            file.write(line)
        file.write(line)
        file.close()





