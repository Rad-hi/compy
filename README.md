# compy

## Task

Given a random pointcloud of shape `(H, W, 4)`, where you have `H * W` 3D points of the form `[X, Y, Z, RGBA]` (made it 4 channels instead of 4 to stay true to the output from something like a [Zed stereo camera](https://www.stereolabs.com/products/zed-x)), how fast can you apply a rotation to all points of the PCL along the X axis ?

## Content of the repo

Welp, what about I provide you with 5 ways to do it, ranging from the simplest pure python solution (`10+ s`), to coding it in C (`5+ ms`), passing by CUDA (`1+ ms`).

## run

### pre-requisites

```bash
python3 -m pip install -r requirements.txt

cd scripts

chmod a+x build.sh

./build.sh
```

### options

```bash
python3 scripts/main.py -h
```

### simplest

```bash
python3 scripts/main.py
```

