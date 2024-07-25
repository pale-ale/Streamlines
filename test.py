import math
import numpy as np
from numpy import ndarray
import scipy.ndimage
import time
from dataclasses import dataclass


a = np.array(
    [
        [0.1,0],
        [0.2,0],
        [0.3,0],
        [0.4,0],
        [0.5,0],
        [0.6,0],
        [0.7,0],
        [0.8,0],
        [0.9,0],
        [1.0,0],
    ], dtype=float
)

class Foo():
    _offset = 0
    _frequency = 1
    _magnitude = 1
    _xspeed = 1
    _lcutoff = .3
    _rcutoff = .6

def get_vector(self, ps: ndarray, t:float) -> ndarray:
        vecs = np.zeros_like(ps)
        period_len = self._frequency * 2 * np.pi
        xvals = self._xspeed
        yvals = np.sin((ps[:,0] + self._offset + t) * period_len) * self._magnitude
        vecs[:,0] = xvals
        vecs[:,1] = np.where( (self._lcutoff < ps[:,0]) & (ps[:,0] < self._rcutoff), yvals, 0)
        return vecs

# print(get_vector(Foo(), a, 0))
#print( (.1 < a[:,0]) & (a[:,0] <.3))

datastr = r"""
[600.738, 606.401, 647.542, 683.247, 700.14, 747.113, 776.618, 846.143, 870.048, 908.393, 928.066, 929.862, 929.862, 929.862, 929.862, 929.862, 929.862, 929.862, 929.862, 929.862, 929.862, 929.862, 929.862, 929.862, 931.591, 931.591, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 932.786, 934.333, 936.797, 936.797, 949.954, 963.133, 963.133, 963.133, 986.039, 986.039, 986.039, 986.039, 986.039, 986.039, 986.039, 986.039, 986.039, 988.596, 988.596, 988.596, 988.596, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078, 989.078]
"""


def listconvert(s:str):
    s = s.strip().replace("[", "").replace("]", "").replace("\n", "").replace(",", "") .strip()
    ys = s.split()
    xs = [x for x in range(len(ys))]
    coords = [f"({x}, {y})" for x,y in zip(xs, ys)]
    result = "\n".join(coords)
    return result

print(listconvert(datastr))
