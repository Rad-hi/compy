import time
import math
import numpy as np
from numba import cuda
import numba

from typing import Union, Tuple, Callable, TypeVar, Optional
from typing_extensions import ParamSpec
T = TypeVar('T')
P = ParamSpec('P')
RotateCallable = Callable[[float, float, float, float], Tuple[float, float]]


X = 0
Y = 1
Z = 2


class TimeMe:
    '''
    Timing decorator that takes into account whether the timing would be done in GPU or CPU
    '''
    def __init__(self, gpu: bool = False, tf: Optional[Callable[[float], float]] = None) -> None:
        self._gpu: bool = gpu
        if tf is None:
            tf = lambda x: x
        self._tf = tf

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Tuple[float, T]:

            ts: float = time.perf_counter()
            result = func(*args, **kwargs)

            # Wait for all jobs in the stream to finish
            if self._gpu:
                cuda.synchronize()

            all_ts = self._tf(time.perf_counter() - ts)
            return (all_ts, result)

        return wrapper


def sec_to_ms(x: float) -> float:
    return x * 1000.0


def rotate_yz_sin_cos(y: float, z: float, sin_: float, cos_: float) -> Tuple[float, float]:
    if math.isfinite(y) and math.isfinite(z):
        yy = y * cos_ - z * sin_
        zz = y * sin_ + z * cos_
        return yy, zz
    return math.nan, math.nan


@numba.njit(cache=True)
def almost_eq(a: Union[int, float], b: Union[int, float], e: Union[int, float]) -> bool: 
    if not (math.isfinite(a) and math.isfinite(b)):
        # If either is nan, consider them equal
        return True
    return abs(a - b) <= e


@numba.njit(nogil=True, cache=True)
def is_almost_eq(mat1: np.ndarray, mat2: np.ndarray, eupsilon: float) -> None:
    '''
    This method is needed (instead of np.all(mat1 == mat2)) becauce
    np.nan == np.nan >>> False. Thus we need to manually check the elements.
    '''
    sh1 = mat1.shape
    sh2 = mat2.shape
    assert sh1 == sh2
    assert mat1.size == mat2.size

    h, w, _ = sh1

    for j in range(h):
        for i in range(w):
            x_eq = almost_eq(mat1[j][i][X], mat2[j][i][X], eupsilon)
            y_eq = almost_eq(mat1[j][i][Y], mat2[j][i][Y], eupsilon)
            z_eq = almost_eq(mat1[j][i][Z], mat2[j][i][Z], eupsilon)
            if not (x_eq and y_eq and z_eq):
                return False
    return True


def get_noise_pcl(shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return np.random.rand(*shape).astype(dtype)
