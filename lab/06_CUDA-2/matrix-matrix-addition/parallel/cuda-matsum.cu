#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "hpc.h"

#define BLKDIM 32

__global__ void matsum_step(float *p, float *q, float *r, int n) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        r[i * n + j] = p[i * n + j] + q[i * n + j];
    }
}

void matsum(float *p, float *q, float *r, int n)
{
    float *d_p, *d_q, *d_r;
    const size_t size = n * n * sizeof(*p);

    cudaSafeCall(cudaMalloc((void **) &d_p, size));
    cudaSafeCall(cudaMalloc((void **) &d_q, size));
    cudaSafeCall(cudaMalloc((void **) &d_r, size));

    cudaSafeCall(cudaMemcpy(d_p, p, size, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_q, q, size, cudaMemcpyHostToDevice));

    dim3 block(BLKDIM, BLKDIM);
    dim3 grid((n + BLKDIM - 1) / BLKDIM, (n + BLKDIM - 1) / BLKDIM);

    matsum_step<<<grid, block>>>(d_p, d_q, d_r, n);
    cudaCheckError();

    cudaSafeCall(cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost));

    cudaSafeCall(cudaFree(d_p));
    cudaSafeCall(cudaFree(d_q));
    cudaSafeCall(cudaFree(d_r));
}

/* Initialize square matrix p of size nxn */
void fill(float *p, int n)
{
    int k = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            p[i * n + j] = k;
            k = (k + 1) % 1000;
        }
    }
}

/* Check result */
int check(float *r, int n)
{
    int k = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabsf(r[i * n + j] - 2.0 * k) > 1e-5) {
                fprintf(stderr, "Check FAILED: r[%d][%d] = %f, expeted %f\n", i, j, r[i * n + j], 2.0 * k);
                return 0;
            }
            k = (k + 1) % 1000;
        }
    }
    printf("Check OK\n");
    return 1;
}

int main(int argc, char *argv[])
{
    float *p, *q, *r;
    int n = 1024;
    const int max_n = 5000;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (n > max_n) {
        fprintf(stderr, "FATAL: the maximum allowed matrix size is %d\n", max_n);
        return EXIT_FAILURE;
    }

    printf("Matrix size: %d x %d\n", n, n);

    const size_t size = n * n * sizeof(*p);

    /* Allocate space for p, q, r */
    p = (float *) malloc(size);
    assert(p != NULL);
    fill(p, n);
    q = (float *) malloc(size);
    assert(q != NULL);
    fill(q, n);
    r = (float *) malloc(size);
    assert(r != NULL);

    const double tstart = hpc_gettime();
    matsum(p, q, r, n);
    const double elapsed = hpc_gettime() - tstart;

    printf("Elapsed time (including data movement): %f\n", elapsed);
    printf("Throughput (Melements/s): %f\n", n * n / (1e6 * elapsed));

    /* Check result */
    check(r, n);

    /* Cleanup */
    free(p);
    free(q);
    free(r);

    return EXIT_SUCCESS;
}
