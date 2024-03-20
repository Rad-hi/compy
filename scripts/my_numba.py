import math

import numpy as np
import numba

from utils import (
    RotateCallable,
    rotate_yz_sin_cos,
    TimeMe, sec_to_ms,
    Y, Z
)


# Jit compile the function to be used by numba
rotate_yz_sin_cos_cpu: RotateCallable = numba.njit(cache=True)(rotate_yz_sin_cos)


'''
NOTE: Beware of the option [fastmath], because it lowers the strictness in regard in how
numba checks against float values
Ref: https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#fastmath
'''
@TimeMe(tf=sec_to_ms)
@numba.njit(cache=True, parallel=True, fastmath=True)
def jit_rotate_pcl_cpu(pcl: np.ndarray, rot: float) -> None:
    sin_ = math.sin(rot)
    cos_ = math.cos(rot)
    h, w, _ = pcl.shape
    for i in numba.prange(h):
        for j in numba.prange(w):
            pcl[i][j][Y], pcl[i][j][Z] = rotate_yz_sin_cos_cpu(
                pcl[i][j][Y], pcl[i][j][Z], sin_, cos_)
