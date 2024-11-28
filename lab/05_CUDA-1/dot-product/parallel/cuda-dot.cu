#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "hpc.h"

#define BLKDIM 1024

__global__ void dot_step(const float *x, const float *y, int n, float *result)
{
    __shared__ float tmp[BLKDIM];
    const int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) {
        tmp[tid] += x[i] * y[i];
    }
    __syncthreads();
    if (tid == 0) {
        *result = 0;
        for (int i = 0; i < blockDim.x; i++) {
            *result += tmp[i];
        }
    }
}

float dot(const float *x, const float *y, int n)
{
    /* Define a `float` variabile `result` in host memory */
    float result;

    /* Allocate space for device copies of `x`, `y` and `result` */
    const size_t SIZE = n * sizeof(float);
    float *d_x;
    float *d_y;
    float *d_result;

    cudaSafeCall(cudaMalloc((void **) &d_x, SIZE));
    cudaSafeCall(cudaMalloc((void **) &d_y, SIZE));
    cudaSafeCall(cudaMalloc((void **) &d_result, sizeof(float)));

    /* Copy `x`, `y` from host to device */
    cudaSafeCall(cudaMemcpy(d_x, x, SIZE, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_y, y, SIZE, cudaMemcpyHostToDevice));

    /* Launch a suitable kernel on the GPU */
    dot_step<<<1, BLKDIM>>>(d_x, d_y, n, d_result);
    cudaCheckError();

    /* Copy the value of `result` back to host memory */
    cudaSafeCall(cudaMemcpy(&result, d_result, sizeof(result), cudaMemcpyDeviceToHost));

    /* Perform the final reduction on the CPU */

    /* Free device memory */
    cudaSafeCall(cudaFree(d_x));
    cudaSafeCall(cudaFree(d_y));
    cudaSafeCall(cudaFree(d_result));

    return result;
}

/**
 * Initialize `x` and `y` of length `n`; return the expected dot
 * product of `x` and `y`. To avoid numerical issues, the vectors are
 * initialized with integer values, so that the result can be computed
 * exactly (save for possible overflows, which should not happen
 * unless the vectors are very long).
 */
float vec_init(float *x, float *y, int n)
{
    const float tx[] = {1, 2, -5};
    const float ty[] = {1, 2, 1};

    const size_t LEN = sizeof(tx) / sizeof(tx[0]);
    const float expected[] = {0, 1, 5};

    for (int i = 0; i < n; i++) {
        x[i] = tx[i % LEN];
        y[i] = ty[i % LEN];
    }

    return expected[n % LEN];
}

int main(int argc, char* argv[])
{
    float *x, *y, result;
    int n = 1024 * 1024;
    const int MAX_N = 128 * n;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [len]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if ((n < 0) || (n > MAX_N)) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n * sizeof(*x);

    /* Allocate space for host copies of x, y */
    x = (float *) malloc(SIZE);
    assert(x != NULL);
    y = (float *) malloc(SIZE);
    assert(y != NULL);
    const float expected = vec_init(x, y, n);

    printf("Computing the dot product of %d elements...\n", n);
    result = dot(x, y, n);
    printf("got=%f, expected=%f\n", result, expected);

    /* Check result */
    if (fabs(result - expected) < 1e-5) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED\n");
    }

    /* Cleanup */
    free(x);
    free(y);
    return EXIT_SUCCESS;
}
