import math

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from utils import (
    RotateCallable,
    rotate_yz_sin_cos,
    TimeMe, sec_to_ms,
    Y, Z
)

from typing import Union

# Jit compile this function as a device function, to be used by the kernel
rotate_yz_sin_cos_gpu: RotateCallable = cuda.jit(device=True, cache=True)(rotate_yz_sin_cos)


@cuda.jit(cache=True)
def rotate_pcl_kernel(pcl: DeviceNDArray, rot: float) -> DeviceNDArray:
    '''
    GPU kernel to rotate pointclouds with an angle rot
    Ref: https://github.com/harrism/numba_examples/blob/master/mandelbrot_numba.ipynb
    '''
    sin_ = math.sin(rot)
    cos_ = math.cos(rot)
    h, w, _ = pcl.shape

    # Figure out where we're at within the GPU grid
    start_x, start_y = cuda.grid(2)
    grid_x = cuda.gridDim.x * cuda.blockDim.x
    grid_y = cuda.gridDim.y * cuda.blockDim.y

    for i in range(start_y, h, grid_y):
        for j in range(start_x, w, grid_x):
            pcl[i][j][Y], pcl[i][j][Z] = rotate_yz_sin_cos_gpu(
                pcl[i][j][Y], pcl[i][j][Z], sin_, cos_)


@TimeMe(gpu=True, tf=sec_to_ms)
def rotate_pcl_gpu(pcl: Union[DeviceNDArray, np.ndarray],
                   rot: float,
                   h2d: bool = False
                   ) -> Union[DeviceNDArray, np.ndarray]:
    # Modify these based on your specific hardware
    blockdim = (32, 16)
    griddim = (32, 16)

    pcl = cuda.to_device(pcl) if h2d else pcl
    rotate_pcl_kernel[griddim, blockdim](pcl, rot)
    pcl = pcl.copy_to_host() if h2d else pcl

    # The rotation doesn't happen inplace only when H2D/D2H
    # transfers were requested; pcl originally on host memory 
    return pcl


def h2d(pcl_gpu: DeviceNDArray) -> np.ndarray:
    return cuda.to_device(pcl_gpu)
