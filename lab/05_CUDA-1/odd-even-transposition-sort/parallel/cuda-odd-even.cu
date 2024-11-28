#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

#define BLKDIM 1024

/* if *a > *b, swap them. Otherwise do nothing */
__host__ __device__ void cmp_and_swap(int *a, int *b)
{
    if (*a > *b) {
        int tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

__global__ void odd_even_step(int *x, int n, int phase)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
#if 0
    if (idx < n - 1) {
        if (idx % 2 == 0) {
            if (phase % 2 == 0) {
                cmp_and_swap(&x[idx], &x[idx + 1]); // = cmp_and_swap(x + idx, x + idx + 1);
            }
        }
        if (idx % 2 == 1) {
            if (phase % 2 == 1) {
                cmp_and_swap(&x[idx], &x[idx + 1]); // = cmp_and_swap(x + idx, x + idx + 1);
            }
        }
    }
#else
    if (idx < n - 1 && idx % 2 == phase % 2) {
        cmp_and_swap(&x[idx], &x[idx + 1]); // = cmp_and_swap(x + idx, x + idx + 1);
    }
#endif
}

/* Odd-even transposition sort */
void odd_even_sort(int *v, int n)
{
#if 0
    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            /* (even, odd) comparisons */
            for (int i = 0; i < n - 1; i += 2) {
                cmp_and_swap(&v[i], &v[i + 1]);
            }
        } else {
            /* (odd, even) comparisons */
            for (int i = 1; i < n - 1; i += 2) {
                cmp_and_swap(&v[i], &v[i + 1]);
            }
        }
    }
#else
    const size_t SIZE = n * sizeof(*v);
    const int NBLOCKS = (n + BLKDIM - 2) / BLKDIM; // I need n - 1 threads
    int *d_v;

    /* We allocate a copy of the array v */
    cudaSafeCall(cudaMalloc((void **) &d_v, SIZE));

    /* We copy v from host to device */
    cudaSafeCall(cudaMemcpy(d_v, v, SIZE, cudaMemcpyHostToDevice));

    for (int phase = 0; phase < n; phase++) {
        odd_even_step<<<NBLOCKS, BLKDIM>>>(d_v, n, phase);
        // Put here because this call inside cudaCheckError() could not happen in case of
        // NO_CUDA_CHECK_ERROR marco definition with -D when compiling
        cudaSafeCall(cudaDeviceSynchronize());
        cudaCheckError(); // To synchronize the execution
    }

    /* We copy d_v from device to host */
    cudaSafeCall(cudaMemcpy(v, d_v, SIZE, cudaMemcpyDeviceToHost));

    /* We free device memory */
    cudaSafeCall(cudaFree(d_v));
#endif
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
    odd_even_sort(x, n);
    elapsed = hpc_gettime() - tstart;
    printf("Sorted %d elements in %f seconds\n", n, elapsed);

    /* Check result */
    check(x, n);

    /* Cleanup */
    free(x);
    return EXIT_SUCCESS;
}
