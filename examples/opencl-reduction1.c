/****************************************************************************
 *
 * opencl-reduction1.c - Reduction with OpenCL
 *
 * This version uses multiple threads of the same workgroup to compute
 * the local reduction. The array length n must be a multiple of
 * BLKDIM, and BLKDIM must be a power of two.
 *
 * Copyright (C) 2019, 2021 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * This program realizes a not-so-simple sum-reduction on the
 * GPU. Each thread block copies a portion of the array in local
 * memory; then, all work-items within the same workgroup cooperate to
 * compute the sum of the local data by organizing the computation as
 * a tree. The final reduction is executed on the CPU.
 *
 * Compile with:
 * cc opencl-reduction1.c simpleCL.c -o opencl-reduction1 -lOpenCL
 *
 * Run with:
 * ./opencl-reduction1
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

int main( void )
{
    sclInitFromFile("opencl-reduction1.cl");

    const int N_OF_BLOCKS = 1024;
    const int N = SCL_DEFAULT_WG_SIZE * N_OF_BLOCKS;
    int *h_a;
    cl_mem d_a, d_sums;
    int h_sums[N_OF_BLOCKS];

    assert( 0 == N % SCL_DEFAULT_WG_SIZE ); /* N must be a multiple of MAX_WG_SIZE */
    assert( (SCL_DEFAULT_WG_SIZE & (SCL_DEFAULT_WG_SIZE-1) ) == 0 ); /* check if the default group size is a power of two using the "bit hack" from http://www.graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2 */

    /* Allocate space for host copies of h_a */
    h_a = (int*)malloc(N * sizeof(int));
    /* Set all elements of vector h_a to 2, so that we know that the
       result of the sum must be 2*N */
    for (int i=0; i<N; i++) { h_a[i] = 2; }
    /* Allocate space for device copy of d_a */
    d_a = sclMallocCopy(N*sizeof(int), h_a, CL_MEM_READ_ONLY);
    d_sums = sclMalloc(sizeof(h_sums), CL_MEM_WRITE_ONLY);
    /* Launch sum() kernel on the GPU */
    sclSetArgsEnqueueKernel(sclCreateKernel("sum_kernel"),
                            DIM1(N), DIM1(SCL_DEFAULT_WG_SIZE),
                            ":b :d :b :l",
                            d_a, N, d_sums, sizeof(h_sums));
    /* Copy the d_sums[] array from device memory to host memory h_sums[] */
    sclMemcpyDeviceToHost(h_sums, d_sums, sizeof(h_sums));
    /* Perform the final reduction on the CPU */
    int s = 0;
    for (int i=0; i<N_OF_BLOCKS; i++) {
        s += h_sums[i];
    }
    /* Check result */
    if ( s != 2*N ) {
        printf("Check FAILED: expected %d, got %d\n", 2*N, s);
    } else {
        printf("Check OK: sum = %d\n", s);
    }
    /* Cleanup */
    free(h_a);
    sclFree(d_a); sclFree(d_sums);
    sclFinalize();
    return EXIT_SUCCESS;
}
