#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

#define BLKDIM 1024

typedef struct {
    float *P;   /* coordinates P[i][j] of point i               */
    int N;      /* Number of points (rows of matrix P)          */
    int D;      /* Number of dimensions (columns of matrix P)   */
} points_t;

/**
 * Read input from stdin. Input format is:
 *
 * d [other ignored stuff]
 * N
 * p0,0 p0,1 ... p0,d-1
 * p1,0 p1,1 ... p1,d-1
 * ...
 * pn-1,0 pn-1,1 ... pn-1,d-1
 *
 */
void read_input(points_t *points)
{
    char buf[1024];
    int N, D;
    float *P;

    if (scanf("%d", &D) != 1) {
        fprintf(stderr, "FATAL: can not read the dimension\n");
        exit(EXIT_FAILURE);
    }
    assert(D >= 2);
    if (NULL == fgets(buf, sizeof(buf), stdin)) { /* ignore rest of the line */
        fprintf(stderr, "FATAL: can not read the first line\n");
        exit(EXIT_FAILURE);
    }
    if (scanf("%d", &N) != 1) {
        fprintf(stderr, "FATAL: can not read the number of points\n");
        exit(EXIT_FAILURE);
    }
    P = (float *) malloc(D * N * sizeof(*P));
    assert(P);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < D; k++) {
            if (1 != scanf("%f", &(P[i * D + k]))) {
                fprintf(stderr, "FATAL: failed to get coordinate %d of point %d\n", k, i);
                exit(EXIT_FAILURE);
            }
        }
    }
    points->P = P;
    points->N = N;
    points->D = D;
}

void free_points(points_t *points)
{
    free(points->P);
    points->P = NULL;
    points->N = points->D = -1;
}

/* Returns 1 if |p| dominates |q| */
__device__ int dominates(const float *const p, const float *const q, const int D)
{
    int greater = 0;
    for (int k = 0; k < D; k++) {
        if (p[k] < q[k]) {
            return 0;
        }
        if (!greater && p[k] > q[k]) {
            greater = 1;
        }
    }
    return greater;
}

__global__ void kernel_skyline_init(int *const s, const int N)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        s[idx] = 1;
    }
}

__global__ void kernel_skyline_step(const float *const P, int *const s, const int N, const int D, int *const r, const int i)
{
    __shared__ int local_r[BLKDIM];
    const int l_index = threadIdx.x;
    const int g_index = threadIdx.x + blockIdx.x * blockDim.x;
    int b_size = blockDim.x / 2;

    local_r[l_index] = 0;

    if (g_index < N) {
        if (s[g_index] && dominates(&(P[i * D]), &(P[g_index * D]), D)) {
            s[g_index] = 0;
            local_r[l_index]--;
        }
    }

    __syncthreads();
    while (b_size > 0) {
        if (l_index < b_size) {
            local_r[l_index] += local_r[l_index + b_size];
        }
        b_size = b_size / 2;
        __syncthreads();
    }

    if (l_index == 0) {
        atomicAdd(r, local_r[0]);
    }
}

/**
 * Compute the skyline of `points`. At the end, `s[i] == 1` if point
 * `i` belongs to the skyline. The function returns the number `r` of
 * points that belongs to the skyline. The caller is responsible for
 * allocating the array `s` of length at least `points->N`.
 */
int skyline(const points_t *points, int *s)
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;
    int r = N;

    const int N_BLOCKS = (N + BLKDIM - 1) / BLKDIM;

    float *d_P;
    int *d_s;
    int *d_r;

    const size_t SIZE_P = N * D * sizeof(*P);
    const size_t SIZE_s = N * sizeof(*s);
    const size_t SIZE_r = sizeof(r);

    cudaSafeCall(cudaMalloc((void **) &d_P, SIZE_P));
    cudaSafeCall(cudaMalloc((void **) &d_s, SIZE_s));
    cudaSafeCall(cudaMalloc((void **) &d_r, SIZE_r));

    cudaSafeCall(cudaMemcpy(d_P, P, SIZE_P, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_r, &r, SIZE_r, cudaMemcpyHostToDevice));

    kernel_skyline_init<<<N_BLOCKS, BLKDIM>>>(d_s, N);
    cudaCheckError();
#ifdef NO_CUDA_CHECK_ERROR // synchronization is always needed
    cudaDeviceSynchronize();
#endif

    kernel_skyline_step<<<N_BLOCKS, BLKDIM>>>(d_P, d_s, N, D, d_r, 0);
    cudaCheckError();

    int is_skyline;
    for (int i = 1; i < N; i++) {
        cudaSafeCall(cudaMemcpy(&is_skyline, &d_s[i], sizeof(is_skyline), cudaMemcpyDeviceToHost));
        if (is_skyline) {
            kernel_skyline_step<<<N_BLOCKS, BLKDIM>>>(d_P, d_s, N, D, d_r, i);
            cudaCheckError();
        }
    }

    cudaSafeCall(cudaMemcpy(s, d_s, SIZE_s, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(&r, d_r, SIZE_r, cudaMemcpyDeviceToHost));

    cudaSafeCall(cudaFree(d_P));
    cudaSafeCall(cudaFree(d_s));
    cudaSafeCall(cudaFree(d_r));

    return r;
}

/**
 * Print the coordinates of points belonging to the skyline `s` to
 * standard output. `s[i] == 1` iff point `i` belongs to the skyline.
 * The output format is the same as the input format, so that this
 * program can process its own output.
 */
void print_skyline(const points_t *points, const int *s, int r)
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;

    printf("%d\n", D);
    printf("%d\n", r);
    for (int i = 0; i < N; i++) {
        if (s[i]) {
            for (int k = 0; k < D; k++) {
                printf("%f ", P[i * D + k]);
            }
            printf("\n");
        }
    }
}

int main(int argc, char *argv[])
{
    points_t points;

    if (argc != 1) {
        fprintf(stderr, "Usage: %s < input_file > output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    read_input(&points);
    int *s = (int *) malloc(points.N * sizeof(*s));
    assert(s);
    const double tstart = hpc_gettime();
    const int r = skyline(&points, s);
    const double elapsed = hpc_gettime() - tstart;
    print_skyline(&points, s, r);

    fprintf(stderr, "\n\t%d points\n", points.N);
    fprintf(stderr, "\t%d dimensions\n", points.D);
    fprintf(stderr, "\t%d points in skyline\n\n", r);
    fprintf(stderr, "Execution time (s) %.2f\n", elapsed);

    free_points(&points);
    free(s);
    return EXIT_SUCCESS;
}
