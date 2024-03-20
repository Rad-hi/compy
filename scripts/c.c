/*
    Rotate pcl C implementation
*/

#include <math.h>
#include <pthread.h>

#define NTHREADS  4  // Number of threads to spawn
#define Y         1
#define Z         2


typedef struct {
    float* head;
    size_t sz;
    size_t stride;
    float cos_;
    float sin_;
} rot_params_t;


static void rot_yz_cos_sin(float* y, float* z, float cos_, float sin_) {
    if (isfinite(*y) && isfinite(*z)) {
        float _y = *y;
        float _z = *z;
        *y = _y * cos_ - _z * sin_;
        *z = _y * sin_ + _z * cos_;
    }
}


/*
    Apply a rotation TF to all components of the contiguous input array.
*/
void rotate_ptc(float* arr, size_t w, size_t h, size_t d, float rot) {
    float cos_ = cos(rot);
    float sin_ = sin(rot);
    for(size_t i = 0; i < w * h * d; i += d) {
        rot_yz_cos_sin(arr + i + Y, arr + i + Z, cos_, sin_);
    }
}


static void* rotate_segment_thread_fn(void* arg) {
    rot_params_t* array = (rot_params_t*)arg;
    for(float* head = array->head; head < array->head + array->sz; head += array->stride) {
        rot_yz_cos_sin(head + Y, head + Z, array->cos_, array->sin_);
    }
    pthread_exit(NULL);
    return NULL;
}


/*
    Apply a rotation TF to all components of the contiguous input array in parallel.
    The way the processing is multithreaded on the array is (expl: nthreads == 4):

                          <----- size ----->
    arr: |****************.****************.****************.****************|
         ^                ^                ^                ^
       slice 0          slice 1          slice 2          slice 3
    
    Each thread will be given a pointer to where the slice starts, and its size.
*/
void rotate_ptc_parallel(float* arr, size_t w, size_t h, size_t d, float rot) {
    static pthread_t threads[NTHREADS];
    static rot_params_t slices[NTHREADS];

    float cos_ = cos(rot);
    float sin_ = sin(rot);

    size_t sz = w * h * d;
    size_t slice_sz = sz / NTHREADS;

    for (size_t i = 0; i < NTHREADS; i++) {
        slices[i].head = arr + i * slice_sz;
        slices[i].sz = slice_sz + (i == NTHREADS - 1 ? sz % NTHREADS : 0);
        slices[i].stride = d;
        slices[i].cos_ = cos_;
        slices[i].sin_ = sin_;
        pthread_create(&threads[i], NULL, rotate_segment_thread_fn, (void*)&slices[i]);
    }

    for (size_t i = 0; i < NTHREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}