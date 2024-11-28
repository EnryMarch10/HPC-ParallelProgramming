#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "hpc.h"

#define BLKDIM 1024

__global__ void reverse_step(int *in, int *out, int n) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        const int opp = n - idx - 1;
        out[opp] = in[idx];
    }
}

/* Reverses `in[]` into `out[]`; assume that `in[]` and `out[]` do not overlap. */
void reverse(int *in, int *out, int n)
{
    const size_t SIZE = n * sizeof(int);
    const int NBLOCKS = (n + BLKDIM - 1) / BLKDIM;
    int *d_in, *d_out;

    cudaSafeCall(cudaMalloc((void **) &d_in, SIZE));
    cudaSafeCall(cudaMalloc((void **) &d_out, SIZE));
    cudaSafeCall(cudaMemcpy(d_in, in, SIZE, cudaMemcpyHostToDevice));
    reverse_step<<<NBLOCKS, BLKDIM>>>(d_in, d_out, n);
    cudaCheckError();
    cudaSafeCall(cudaMemcpy(out, d_out, SIZE, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaFree(d_in));
    cudaSafeCall(cudaFree(d_out));
}

__global__ void inplace_reverse_step(int *in, int n) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n / 2) {
        const int opp = n - idx - 1;
        int tmp = in[idx];
        in[idx] = in[opp];
        in[opp] = tmp;
    }
}

/* In-place reversal of in[] into itself. */
void inplace_reverse(int *in, int n)
{
    const size_t SIZE = n * sizeof(int);
    const int NBLOCKS = (n / 2 + BLKDIM - 1) / BLKDIM;
    int *d_in;

    cudaSafeCall(cudaMalloc((void **) &d_in, SIZE));
    cudaSafeCall(cudaMemcpy(d_in, in, SIZE, cudaMemcpyHostToDevice));
    inplace_reverse_step<<<NBLOCKS, BLKDIM>>>(d_in, n);
    cudaCheckError();
    cudaSafeCall(cudaMemcpy(in, d_in, SIZE, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaFree(d_in));
}

void fill(int *x, int n)
{
    for (int i = 0; i < n; i++) {
        x[i] = i;
    }
}

int check(const int *x, int n)
{
    for (int i = 0; i < n; i++) {
        if (x[i] != n - 1 - i) {
            fprintf(stderr, "Test FAILED: x[%d]=%d, expected %d\n", i, x[i], n - 1 - i);
            return 0;
        }
    }
    printf("Test OK\n");
    return 1;
}

int main(int argc, char *argv[])
{
    int *in, *out;
    int n = 1024 * 1024;
    const int MAX_N = 512 * 1024 * 1024;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (n > MAX_N) {
        fprintf(stderr, "FATAL: input too large (maximum allowed length is %d)\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n * sizeof(*in);

    /* Allocate in[] and out[] */
    in = (int *) malloc(SIZE);
    assert(in != NULL);
    out = (int *) malloc(SIZE);
    assert(out != NULL);
    fill(in, n);

    printf("Reverse %d elements... ", n);
    reverse(in, out, n);
    check(out, n);

    printf("In-place reverse %d elements... ", n);
    inplace_reverse(in, n);
    check(in, n);

    /* Cleanup */
    free(in);
    free(out);

    return EXIT_SUCCESS;
}
