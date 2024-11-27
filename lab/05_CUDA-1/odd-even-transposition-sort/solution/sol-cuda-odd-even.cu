/****************************************************************************
 *
 * cuda-odd-even.cu - Odd-even sort
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
#include <assert.h>

#include "hpc.h"

#define BLKDIM 1024

/* if *a > *b, swap them. Otherwise do nothing */
__device__ void cmp_and_swap(int* a, int* b)
{
    if (*a > *b) {
        int tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

/**
 * This kernel requires `n` threads to sort `n` elements, but only
 * half the threads are used during each phase. Therefore, this kernel
 * is not efficient.
 */
__global__ void odd_even_step_bad(int *x, int n, int phase)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n - 1 && idx % 2 == phase % 2) {
        /* Compare & swap x[idx] and x[idx+1] */
        cmp_and_swap(&x[idx], &x[idx + 1]);
    }
}

/* Odd-even transposition sort */
void odd_even_sort_bad(int *v, int n)
{
    int *d_v; /* device copy of `v` */
    const int NBLOCKS = (n + BLKDIM - 1) / BLKDIM;
    const size_t SIZE = n * sizeof(*d_v);

    /* Allocate `d_v` on device */
    cudaSafeCall(cudaMalloc((void **) &d_v, SIZE));

    /* Copy `v` to device memory */
    cudaSafeCall(cudaMemcpy(d_v, v, SIZE, cudaMemcpyHostToDevice));

    printf("BAD version (%d elements, %d CUDA threads):\n", n, NBLOCKS * BLKDIM);
    for (int phase = 0; phase < n; phase++) {
        odd_even_step_bad<<<NBLOCKS, BLKDIM>>>(d_v, n, phase);
        cudaCheckError();
    }

    /* Copy result back to host */
    cudaSafeCall(cudaMemcpy(v, d_v, SIZE, cudaMemcpyDeviceToHost));

    /* Free memory on the device */
    cudaSafeCall(cudaFree(d_v));
}

/**
 * A more efficient kernel that uses n/2 threads to sort n elements.
 */
__global__ void odd_even_step_good(int *x, int n, int phase)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x; /* thread index */
    const int idx = tid * 2 + (phase % 2); /* array index handled by this thread */
    if (idx < n - 1) {
        cmp_and_swap(&x[idx], &x[idx + 1]);
    }
}

/* This function is almost identical to odd_even_sort_bad(), with the
   difference that it uses a more efficient kernel
   (odd_even_step_good()) that only requires n/2 threads during each
   phase. */
void odd_even_sort_good(int *v, int n)
{
    int *d_v; /* device copy of v */
    const int NBLOCKS = (n / 2 + BLKDIM - 1) / BLKDIM;
    const size_t SIZE = n * sizeof(*d_v);

    /* Allocate d_v on device */
    cudaSafeCall(cudaMalloc((void **) &d_v, SIZE));

    /* Copy v to device memory */
    cudaSafeCall(cudaMemcpy(d_v, v, SIZE, cudaMemcpyHostToDevice));

    printf("GOOD version (%d elements, %d CUDA threads):\n", n, NBLOCKS * BLKDIM);
    for (int phase = 0; phase < n; phase++) {
        odd_even_step_good<<<NBLOCKS, BLKDIM>>>(d_v, n, phase);
        cudaCheckError();
    }

    /* Copy result back to host */
    cudaSafeCall(cudaMemcpy(v, d_v, SIZE, cudaMemcpyDeviceToHost));

    /* Free memory on the device */
    cudaSafeCall(cudaFree(d_v));
}

/**
 * Return a random integer in the range [a..b]
 */
int randab(int a, int b)
{
    return a + (rand() % (b - a + 1));
}

/**
 * Fill vector x with a random permutation of the integers 0..n-1
 */
void fill(int *x, int n)
{
    for (int i = 0; i < n; i++) {
        x[i] = i;
    }
    for(int i = 0; i < n - 1; i++) {
        const int j = randab(i, n - 1);
        const int tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }
}

/**
 * Check correctness of the result
 */
int check(const int *x, int n)
{
    for (int i = 0; i < n; i++) {
        if (x[i] != i) {
            fprintf(stderr, "Check FAILED: x[%d]=%d, expected %d\n", i, x[i], i);
            return 0;
        }
    }
    printf("Check OK\n");
    return 1;
}

int main(int argc, char *argv[])
{
    int *x;
    int n = 128 * 1024;
    const int MAX_N = 512 * 1024 * 1024;
    double tstart, elapsed;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [len]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (n > MAX_N) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n * sizeof(*x);

    /* Allocate space for x on host */
    x = (int *) malloc(SIZE);
    assert(x != NULL);
    fill(x, n);

    tstart = hpc_gettime();
    odd_even_sort_bad(x, n);
    elapsed = hpc_gettime() - tstart;
    printf("Sorted %d elements in %f seconds\n", n, elapsed);

    /* Check result */
    check(x, n);

    fill(x, n);

    tstart = hpc_gettime();
    odd_even_sort_good(x, n);
    elapsed = hpc_gettime() - tstart;
    printf("Sorted %d elements in %f seconds\n", n, elapsed);

    check(x, n);

    /* Cleanup */
    free(x);
    return EXIT_SUCCESS;
}
