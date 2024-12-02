/****************************************************************************
 *
 * demo-cuda-tput.cu -- Dense matrix-matrix multiplication with CUDA and OpenMP
 *
 * Copyright (C) 2018, 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * --------------------------------------------------------------------------
 *
 * To compile:
 *
 * nvcc -Xcompiler -fopenmp -O2 demo-cuda-tput.cu -o demo-cuda-tput -lm
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>     /* for malloc() */
#include <math.h>       /* for fabsf()  */
#include <strings.h>    /* for bzero()  */

#include "hpc.h"

#define BLKDIM 32

/* Compute r = p * q; does not require that n is a multiple of
   BLKDIM. To do so, it fills shared buffers so that values outside
   the matrices are treated as zeros. */
__global__ void gpu_matmul(const float *p, const float *q, float *r, int n)
{
    __shared__ float local_p[BLKDIM][BLKDIM];
    __shared__ float local_q[BLKDIM][BLKDIM];
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i = by * BLKDIM + ty;
    const int j = bx * BLKDIM + tx;
    float v = 0.0;
    for (int m = 0; m < n; m += BLKDIM) { /* loop over tiles */
        local_p[ty][tx] = local_q[ty][tx] = 0;
        if (i < n && m + tx < n) {
            local_p[ty][tx] = p[i * n + (m + tx)];
        }
        if (j < n && m + ty < n) {
            local_q[ty][tx] = q[(m + ty) * n + j];
        }
        __syncthreads();
        for (int k = 0; k < BLKDIM; k++) { /* loop within tile */
            v += local_p[ty][k] * local_q[k][tx];
        }
        __syncthreads();
    }
    if (i < n && j < n) {
        r[i*n + j] = v; /* write result to global memory */
    }
}


/* Cache-efficient computation of r = p * q, where p. q, r are n x n
  matrices. The caller is responsible for allocating the memory for
  r. This function allocates (and the frees) an additional n x n
  temporary matrix. */
void cpu_matmul(float *p, float* q, float *r, int n)
{
    float *qT = (float *) malloc(n * n * sizeof(float));

    /* transpose q, storing the result in qT */
#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                qT[j * n + i] = q[i * n + j];
            }
        }

        /* multiply p and qT row-wise */
#pragma omp for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += p[i * n + k] * qT[j * n + k];
                }
                r[i * n + j] = s;
            }
        }
    }

    free(qT);
}

/* Initialize square matrix q */
void mat_init(float *q, int n)
{
    for (int i = 0; i < n * n; i++) {
        q[i] = 1.0;
    }
}

int check_result(const float *r, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabsf(r[i * n + j] - n) > 1e-5) {
                printf("Check failed: r[%d][%d] = %f, expected %f\n", i, j, r[i * n + j], (float) n);
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char *argv[])
{
    float *p, *q, *r; /* host copies of p, q, r */
    float *d_p, *d_q, *d_r; /* device copies of p, q, r */
    int N = 512, nrep = 1;
    double elapsed;

    if (argc > 3) {
        fprintf(stderr, "Usage: %s [n [nrep]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (argc > 2) {
        nrep = atoi(argv[2]);
    }

    dim3 block(BLKDIM, BLKDIM);
    dim3 grid((N + BLKDIM - 1) / BLKDIM, (N + BLKDIM - 1) / BLKDIM);
    const size_t size = N * N * sizeof(float);

    /* Allocate space for device copies of p, q, r */
    cudaMalloc((void **) &d_p, size);
    cudaMalloc((void **) &d_q, size);
    cudaMalloc((void **) &d_r, size);

    /* Allocate space for host copies of p, q, r */
    p = (float *) malloc(size);
    mat_init(p, N);
    q = (float *) malloc(size);
    mat_init(q, N);
    r = (float *) malloc(size);

    /* Copy inputs to device */
    cudaMemcpy(d_p, p, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, size, cudaMemcpyHostToDevice);

    printf("%d\t", N);

    /* I have observed that, on my machine, the first kernel execution
       takes much more time than the subsequent ones. Therefore, to
       get more precise measures, we do a "dummy" run outside of the
       measurement loop. */
    gpu_matmul<<<grid, block>>>(d_p, d_q, d_r, N);
    cudaDeviceSynchronize();

    /**
     ** Matrix-matrix multiply on the GPU.  To perform a fair
     ** comparison with OpenMP, we measure also the time needed to
     ** copy the input to the device, and to copy the result back to
     ** the host.
     **/
    elapsed = 0.0;
    for (int rep = 0; rep < nrep; rep++) {
        const double tstart = hpc_gettime();
        cudaMemcpy(d_p, p, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q, size, cudaMemcpyHostToDevice);
        gpu_matmul<<<grid, block>>>(d_p, d_q, d_r, N);
        cudaDeviceSynchronize();
        cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);
        const double this_elapsed = hpc_gettime() - tstart;
        elapsed += this_elapsed;
    }
    elapsed /= nrep;
    printf("%f\t", elapsed);

    /* Check correctness */
    check_result(r, N);

    /**
     ** Matrix-matrix multiply on the CPU
     **/
    elapsed = 0.0;
    for (int rep = 0; rep < nrep; rep++) {
        const double tstart = hpc_gettime();
        cpu_matmul(p, q, r, N);
        const double this_elapsed = hpc_gettime() - tstart;
        elapsed += this_elapsed;
    }
    elapsed /= nrep;
    printf("%f\n", elapsed);

    /* Cleanup */
    free(p);
    free(q);
    free(r);
    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_r);
    return EXIT_SUCCESS;
}
