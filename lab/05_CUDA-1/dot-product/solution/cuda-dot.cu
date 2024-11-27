/****************************************************************************
 *
 * cuda-dot.cu - Dot product
 *
 * Copyright (C) 2017--2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "hpc.h"

#define BLKDIM 1024

__global__ void dot_kernel(const float *x, const float *y, int n, float *result)
{
    __shared__ float tmp[BLKDIM];
    const int tid = threadIdx.x;
    float s = 0.0;
    for (int i = tid; i < n; i += blockDim.x) {
        s += x[i] * y[i];
    }
    tmp[tid] = s;
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
    float result;
    float *d_x, *d_y, *d_result; /* device copies of x, y, result */
    const size_t SIZE_XY = n * sizeof(*x);
    const size_t SIZE_RESULT = sizeof(result);

    /* Allocate space for device copies of x, y */
    cudaSafeCall(cudaMalloc((void **) &d_x, SIZE_XY));
    cudaSafeCall(cudaMalloc((void **) &d_y, SIZE_XY));
    cudaSafeCall(cudaMalloc((void **) &d_result, SIZE_RESULT));

    /* Copy inputs to device memory */
    cudaSafeCall(cudaMemcpy(d_x, x, SIZE_XY, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_y, y, SIZE_XY, cudaMemcpyHostToDevice));

    /* Launch dot_kernel() on GPU */
    dot_kernel<<<1, BLKDIM>>>(d_x, d_y, n, d_result);
    cudaCheckError();

    /* Copy the result back to host memory */
    cudaSafeCall(cudaMemcpy(&result, d_result, SIZE_RESULT, cudaMemcpyDeviceToHost));

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
