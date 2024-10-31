/****************************************************************************
 *
 * opencl-matmul.c - Dense matrix-matrix multiplication with OpenCL
 *
 * Copyright (C) 2018--2021 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * ---------------------------------------------------------------------------
 *
 * Dense matrix-matrix multiplication kernel with CUDA. Two versions
 * of the kernel are provided: one that does not use shared memory,
 * and one that does. A third version uses shared memory and does not
 * require that the matrix size is multiple of BLKDIM.
 *
 * Compile with:
 * gcc opencl-matmul.c simpleCL.c -o opencl-matmul -lm -lOpenCL
 *
 * Run with:
 * ./opencl-matmul [N]
 *
 ****************************************************************************/

/* The following #define is required by the implementation of
   hpc_gettime(). It MUST be defined before including any other
   file. */
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>     /* for malloc() */
#include <math.h>       /* for fabsf()  */
#include <strings.h>    /* for bzero()  */
#include <assert.h>

#include "simpleCL.h"
#include "hpc.h"

#define BLKDIM 16

/* Initialize square matrix q */
void mat_init( float *q, int n )
{
    for (int i=0; i<n*n; i++) {
        q[i] = 1.0;
    }
}

int check_result( const float *r, int n )
{
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (fabsf(r[i*n+j] - n) > 1e-5) {
                printf("Check failed: r[%d][%d] = %f, expected %f\n", i, j, r[i*n+j], (float)n);
                return 0;
            }
        }
    }
    printf("Check OK\n");
    return 1;
}

size_t roundup(size_t s, size_t m)
{
    return ((s+m-1)/m)*m;
}

int main( int argc, char* argv[] )
{
    float *p, *q, *r;	          /* host copies of p, q, r */
    cl_mem d_p, d_q, d_r;	  /* device copies of p, q, r */
    int N = 512;
    const int NMAX = 4096;
    double tstart, tstop, tnoshared, tshared;

    if ( argc > 1 ) {
        N = atoi(argv[1]);
    }

    assert(N <= NMAX);

    sclInitFromFile("opencl-matmul.cl");
    const sclDim block = DIM2(BLKDIM, BLKDIM);
    const sclDim grid = DIM2(roundup(N, BLKDIM), roundup(N, BLKDIM));
    const size_t size = N*N*sizeof(float);

    /* Allocate space for host copies of p, q, r */
    p = (float*)malloc(size); mat_init(p, N);
    q = (float*)malloc(size); mat_init(q, N);
    r = (float*)malloc(size);

        /* Allocate space for device copies of p, q, r */
    d_p = sclMallocCopy(size, p, CL_MEM_READ_ONLY);
    d_q = sclMallocCopy(size, q, CL_MEM_READ_ONLY);
    d_r = sclMalloc(size, CL_MEM_WRITE_ONLY);

    printf("Matrix-Matrix multiplication (%dx%d)\n", N, N);

    /**
     ** Matrix-matrix multiply WITHOUT shared memory
     **/
    printf("No shared memory:\t");
    sclKernel matmul = sclCreateKernel("matmul");
    tstart = hpc_gettime();
    sclSetArgsLaunchKernel(matmul,
                           grid, block,
                           ":b :b :b :d",
                           d_p,
                           d_q,
                           d_r,
                           N);
    tstop = hpc_gettime();
    tnoshared = tstop - tstart;
    printf("%fs\n", tnoshared);
    /* Copy result back to host and check correctness */
    bzero(r, size); /* erase destination buffer, just in case... */
    sclMemcpyDeviceToHost(r, d_r, size);
    check_result(r, N);

    /* zero out r and d_r, to ensure that we don't read old results */
    sclMemset(d_r, 0, size);
    bzero(r, size);

    /**
     ** Matrix-matrix multiply WITH shared memory
     **/
    printf("Shared memory:\t\t");
    sclKernel matmulb_generic = sclCreateKernel("matmulb_generic");
    tstart = hpc_gettime();
    sclSetArgsLaunchKernel(matmulb_generic,
                           grid, block,
                           ":b :b :b :d",
                           d_p,
                           d_q,
                           d_r,
                           N);
    tstop = hpc_gettime();
    tshared = tstop - tstart;
    printf("%fs (%.2fx speedup)\n", tshared, tnoshared / tshared);
    /* Copy result back to host and check correctness */
    sclMemcpyDeviceToHost(r, d_r, size);
    check_result(r, N);

    /* Cleanup */
    free(p);
    free(q);
    free(r);
    sclFree(d_p);
    sclFree(d_q);
    sclFree(d_r);
    sclReleaseKernel(matmul);
    sclReleaseKernel(matmulb_generic);
    sclFinalize();
    return EXIT_SUCCESS;
}
