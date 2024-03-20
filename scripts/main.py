# This code was presented in this public PR: https://github.com/stereolabs/zed-python-api/pull/230

# task: apply a 69deg rotation on all points in the PCL along the X axis
# (believe me, this is a real world application, it's not just for fun)
import math
import random
import argparse

import numpy as np

from tqdm import tqdm

import c as C
from utils import (
    is_almost_eq,
    get_noise_pcl,
    TimeMe, sec_to_ms,
    rotate_yz_sin_cos,
    Y, Z
)

# NOTE: the gpu variants are only imported when the user specifies GPU

from my_numba import jit_rotate_pcl_cpu

from typing import Tuple, List


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
def c_rotate_pcl_cpu(ptc: np.ndarray, angle: float) -> None:
    C.rotate_ptc(ptc, angle)


@TimeMe(tf=sec_to_ms)
def c_rotate_pcl_cpu_threaded(ptc: np.ndarray, angle: float) -> None:
    C.rotate_ptc_parallel(ptc, angle)


def test_correctness(pcl: np.ndarray, rot: float, eupsilon: float, *, gpu: bool = False) -> None:
    '''
    Test that the 3 available methods are calculating the same values
    (since the numba based one is already validated)
    '''
    pcl_0 = pcl.copy()
    jit_rotate_pcl_cpu(pcl_0, rot)

    pcl_1 = pcl.copy()
    c_rotate_pcl_cpu_threaded(pcl_1, rot)

    pcl_2 = pcl.copy()
    c_rotate_pcl_cpu(pcl_2, rot)

    if gpu:
        pcl_3 = pcl.copy()
        _, pcl_3 = rotate_pcl_gpu(pcl_3, rot, h2d=True)

    # Test that all rotation functions are giving the same results
    assert is_almost_eq(pcl_0, pcl_1, eupsilon)
    assert is_almost_eq(pcl_0, pcl_2, eupsilon)
    assert is_almost_eq(pcl_1, pcl_2, eupsilon)
    if gpu:
        assert is_almost_eq(pcl_0, pcl_3, eupsilon)
        assert is_almost_eq(pcl_1, pcl_3, eupsilon)
        assert is_almost_eq(pcl_2, pcl_3, eupsilon)

    # Test that the rotation actually happened
    assert not is_almost_eq(pcl_0, pcl, eupsilon)
    assert not is_almost_eq(pcl_1, pcl, eupsilon)
    assert not is_almost_eq(pcl_2, pcl, eupsilon)
    if gpu:
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

    print('PCL stats:')
    KILO_B: int = 1024
    for i, sample in enumerate(pcls):
        h, w, c = sample.shape
        print(f'\t{i} - PCL shape: {w = }, {h = }, {c = }, {sample.dtype}, sz = {(sample.size * sample.itemsize) / (KILO_B ** 2):.1f} Mb')

    ROT: float = 69.0
    EUPSILON: float = 0.001
    test_correctness(pcls[0], ROT, EUPSILON, gpu=gpu)

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

    n_pcls = len(pcls)
    for _ in tqdm(range(n_iters)):
        pcl = pcls[random.randint(0, n_pcls - 1)]

        if pure:
            py_rot_cpu += py_rotate_pcl_cpu(pcl.copy(), ROT)[0]

        if gpu:
            ## This takes a loot of time: ~36 ms
            pcl_gpu = h2d(pcl.copy())
            rot_gpu += rotate_pcl_gpu(pcl_gpu, ROT, h2d=False)[0]

        jit_rot_cpu += jit_rotate_pcl_cpu(pcl.copy(), ROT)[0]
        c_rot_cpu_prll += c_rotate_pcl_cpu_threaded(pcl.copy(), ROT)[0]
        c_rot_cpu_ctgs += c_rotate_pcl_cpu(pcl.copy(), ROT)[0]

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
    parser.add_argument('-i', '--iters', required=False, type=int, default=1000, help='Number of iterations')
    parser.add_argument('-n', '--npcl', required=False, type=int, default=5, help='Number of random PCLs to generate')
    parser.add_argument('-p', '--pure', required=False, action='store_true', help='Run the pure python version or not')
    parser.add_argument('-g', '--gpu', required=False, action='store_true', help='Run the gpu version or not')
    parser.add_argument('-r', '--rand', required=False, action='store_false', help='Randomize the shape of the PCL or not')
    args = parser.parse_args()

    if args.gpu:
        # This is imported here so that if the user doesn't have a CUDA capable GPU
        # they can simply not provide the GPU option, and cuda wouldn't even be imported
        from my_cuda import rotate_pcl_gpu, h2d

    # cap it at 2K resolution
    H, W = 1242, 2208
    SHAPE: Tuple[int, int, int] = (H, W, 4)
    if args.rand:
        shapes = [SHAPE] * args.npcl
    else:
        shapes = []
        for _ in range(args.npcl):
            h = random.randint(int(H * 0.8), H)
            h -= h % 2  # make sure it's pair
            w = random.randint(int(W * 0.8), W)
            w -= w % 2
            shapes.append((h, w, 4))

    pcls = [get_noise_pcl(sh, np.float32) for sh in shapes]
    main(pcls, args.iters, pure=args.pure, gpu=args.gpu)
