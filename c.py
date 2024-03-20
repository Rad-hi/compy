'''
This is a wrapper file for the C library functions
Ref: https://stackoverflow.com/a/16647916
Ref: https://numpy.org/doc/stable/reference/routines.ctypeslib.html
'''
import ctypes
import numpy as np
from os.path import dirname, abspath, join

d = dirname(abspath(__file__))
# Load the shared library file (if intended to run on multiple platforms: check and on WIN put .dll instead of .so)
lib = ctypes.CDLL(join(d, 'lib.so'))

# Some types for the function signature definition
C_I32 = ctypes.c_int32
C_F32 = ctypes.c_float
ARR_3D_F32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags='C_CONTIGUOUS')

# Define the function signatures
lib.rotate_ptc.argtypes = [ARR_3D_F32, C_I32, C_I32, C_I32, C_F32]
lib.rotate_ptc_parallel.argtypes = [ARR_3D_F32, C_I32, C_I32, C_I32, C_F32]


def rotate_ptc(arr: np.ndarray, rot: float) -> None:
    h, w, d = arr.shape
    lib.rotate_ptc(arr, C_I32(w), C_I32(h), C_I32(d), rot)

def rotate_ptc_parallel(arr: np.ndarray, rot: float) -> None:
    h, w, d = arr.shape
    lib.rotate_ptc_parallel(arr, C_I32(w), C_I32(h), C_I32(d), rot)
