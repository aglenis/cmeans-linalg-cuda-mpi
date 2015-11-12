// #define TILE_DIM 32
// #define SSIZE_1SHARED 32
// #define BLOCK_SIZE 32
// #define AS(i, j) As[i][j]
// #define BS(i, j) Bs[i][j]
// #define n_TPB %(n_TPB)s
// #define WIDTH 512
#include <math.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#define NDATA 11264
#define NCLASS 512
#define NDIM 512
