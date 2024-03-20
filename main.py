# This code was presented in this public PR: https://github.com/stereolabs/zed-python-api/pull/230

# task: apply a 69deg rotation on all points in the PCL along the X axis
# (believe me, this is a real world application, it's not just for fun)
import math
import random
import argparse

import numpy as np
import numba
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from tqdm import tqdm

import c as C
from utils import *

from typing import Union, Tuple, Callable, List
RotateCallable = Callable[[float, float, float, float], Tuple[float, float]]


# JIT compile for both CPU and GPU
rotate_yz_sin_cos_cpu: RotateCallable = numba.njit(cache=True)(rotate_yz_sin_cos)
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


def sec_to_ms(x: float) -> float:
    return x * 1000.0


@TimeMe(gpu=True, tf=sec_to_ms)
def rotate_pcl_gpu(pcl: Union[DeviceNDArray, np.ndarray],
                   rot: float,
                   h2d: bool = False
                   ) -> Union[DeviceNDArray, np.ndarray]:
    blockdim = (32, 16)
    griddim = (32, 16)

    pcl = cuda.to_device(pcl) if h2d else pcl
    rotate_pcl_kernel[griddim, blockdim](pcl, rot)
    pcl = pcl.copy_to_host() if h2d else pcl

    # The rotation doesn't happen inplace only when H2D/D2H
    # transfers were requested; pcl originally on host memory 
    return pcl


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


@TimeMe(tf=sec_to_ms)
def py_rotate_pcl_cpu(pcl: np.ndarray, rot: float) -> None:
    sin_ = math.sin(rot)
    cos_ = math.cos(rot)
    h, w, _ = pcl.shape
    for i in range(h):
        for j in range(w):
            pcl[i][j][Y], pcl[i][j][Z] = rotate_yz_sin_cos(
                pcl[i][j][Y], pcl[i][j][Z], sin_, cos_)


@TimeMe(tf=sec_to_ms)
def rotate_c(ptc: np.ndarray, angle: float) -> None:
    C.rotate_ptc(ptc, angle)


@TimeMe(tf=sec_to_ms)
def rotate_c_p(ptc: np.ndarray, angle: float) -> None:
    C.rotate_ptc_parallel(ptc, angle)


def test_correctness(pcl: np.ndarray, rot: float, eupsilon: float) -> None:
    '''
    Test that the 3 available methods are calculating the same values
    (since the numba based one is already validated)
    '''
    pcl_0 = pcl.copy()
    jit_rotate_pcl_cpu(pcl_0, rot)

    pcl_1 = pcl.copy()
    rotate_c_p(pcl_1, rot)

    pcl_2 = pcl.copy()
    rotate_c(pcl_2, rot)

    pcl_3 = pcl.copy()
    _, pcl_3 = rotate_pcl_gpu(pcl_3, rot, h2d=True)

    # Test that all rotation functions are giving the same results
    assert is_almost_eq(pcl_0, pcl_1, eupsilon)
    assert is_almost_eq(pcl_0, pcl_2, eupsilon)
    assert is_almost_eq(pcl_0, pcl_3, eupsilon)
    assert is_almost_eq(pcl_1, pcl_2, eupsilon)
    assert is_almost_eq(pcl_1, pcl_3, eupsilon)
    assert is_almost_eq(pcl_2, pcl_3, eupsilon)

    # Test that the rotation actually happened
    assert not is_almost_eq(pcl_0, pcl, eupsilon)
    assert not is_almost_eq(pcl_1, pcl, eupsilon)
    assert not is_almost_eq(pcl_2, pcl, eupsilon)
    assert not is_almost_eq(pcl_3, pcl, eupsilon)

    print("Functions are behaving as expected.")


def warmup_gpu(n_iters: int = 100):
    dummy_input = np.ones((1200, 2202, 4), dtype=np.float32) * random.random()
    for _ in tqdm(range(n_iters)):
        rotate_pcl_gpu(dummy_input, 15.0, h2d=True)


def main(pcls: List[np.ndarray],
         n_iters: int = 1000,
         *,
         pure: bool = False,
         gpu: bool = True,
         ) -> None:

    sample = pcls[0]
    
    KILO_B: int = 1024
    h, w, c = sample.shape
    print(f'PCL shape: {w = }, {h = }, {c = }, {sample.dtype}, sz = {(sample.size * sample.itemsize) // (KILO_B ** 2)} Mb')

    ROT = 69.0
    test_correctness(sample, ROT, 0.000_001)

    if gpu:
        print(f'Warming up the GPU')
        warmup_gpu()

    print('Benchmarking')
    if pure:
        py_rot_cpu: float = 0.0
    if gpu:
        rot_gpu: float = 0.0
    jit_rot_cpu: float = 0.0
    c_rot_cpu_prll: float = 0.0
    c_rot_cpu_ctgs: float = 0.0

    for _ in tqdm(range(n_iters)):
        pcl = pcls[random.randint(0, len(pcls) - 1)]

        if pure:
            py_rot_cpu += py_rotate_pcl_cpu(pcl.copy(), ROT)[0]

        if gpu:
            # This takes a loot of time: ~30 ms
            pcl_gpu = cuda.to_device(pcl.copy())
            rot_gpu += rotate_pcl_gpu(pcl_gpu, ROT)[0]

        jit_rot_cpu += jit_rotate_pcl_cpu(pcl.copy(), ROT)[0]
        c_rot_cpu_prll += rotate_c_p(pcl.copy(), ROT)[0]
        c_rot_cpu_ctgs += rotate_c(pcl.copy(), ROT)[0]

    if pure:
        py_rot_cpu /= n_iters
    if gpu:
        rot_gpu /= n_iters
    jit_rot_cpu /= n_iters
    c_rot_cpu_prll /= n_iters
    c_rot_cpu_ctgs /= n_iters

    msg: str = ''
    msg += f"{jit_rot_cpu = :.1f} ms, "
    msg += f"{c_rot_cpu_prll = :.1f} ms, "
    msg += f"{c_rot_cpu_ctgs = :.1f} ms, "
    if gpu:
        msg += f"{rot_gpu = :.1f} ms, "
    if pure:
        msg += f"{py_rot_cpu = :.1f} ms, "
    print(msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a bemchmark.')
    parser.add_argument('-i', '--iters', required=False, type=int, default=1000, help='Number of iterations to be performed')
    parser.add_argument('-p', '--pure', required=False, action='store_true', help='Whether to run the pure python version or not')
    parser.add_argument('-g', '--gpu', required=False, action='store_true', help='Whether to run the gpu version or not')

    args = parser.parse_args()

    pcl: np.ndarray = np.load('random_pcl/pcl.npy')

    # TODO: randomize the PCL by introducing noise to it
    # create 5-10 versions of it
    main([pcl], args.iters, pure=args.pure, gpu=args.gpu)
