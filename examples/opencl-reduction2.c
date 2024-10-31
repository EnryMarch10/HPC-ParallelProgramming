/****************************************************************************
 *
 * opencl-reduction2.c - Reduction with OpencL
 *
 * This is a version of opencl-reduction1.c that works for any array
 * length n; however, it still requires that BLKDIM is a power of two.
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
 * GPU. Each workgroup copies a portion of the array in local memory;
 * then, all work-items within the same workgroup cooperate to compute
 * the sum of the local data by organizing the computation as a
 * tree. The final reduction is executed on the CPU.
 *
 * Compile with:
 * cc opencl-reduction2.c simpleCL.c -o opencl-reduction2 -lOpenCL
 *
 * Run with:
 * ./opencl-reduction2
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

void init( int *v, int n )
{
    int i;
    for (i=0; i<n; i++) {
        v[i] = 2;
    }
}

int main( int argc, char *argv[] )
{
    sclInitFromFile("opencl-reduction2.cl");

    int *h_a, *h_out;
    cl_mem d_a, d_out;
    int n = 8192;

    assert( (SCL_DEFAULT_WG_SIZE & (SCL_DEFAULT_WG_SIZE-1)) == 0 ); /* check if the default group size is a power of two using the "bit hack" from http://www.graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2 */

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    const size_t SIZE = n * sizeof(*h_a);
    const int N_OF_GROUPS = (n + SCL_DEFAULT_WG_SIZE - 1)/SCL_DEFAULT_WG_SIZE;
    const int SIZE_OUT = N_OF_GROUPS * sizeof(*h_out);

    /* Allocate space for host copies of a[] and tmp[] */
    h_a = (int*)malloc(SIZE); assert(h_a != NULL);
    h_out = (int*)malloc(SIZE_OUT); assert(h_out != NULL);
    init(h_a, n);

    /* Allocate space for device copies of d_a */
    d_a = sclMallocCopy(SIZE, h_a, CL_MEM_READ_ONLY);
    d_out = sclMalloc(SIZE_OUT, CL_MEM_WRITE_ONLY);

    /* Launch sum() kernel on the GPU */
    sclSetArgsEnqueueKernel(sclCreateKernel("sum_kernel"),
                            DIM1(N_OF_GROUPS * SCL_DEFAULT_WG_SIZE), DIM1(SCL_DEFAULT_WG_SIZE),
                            ":b :d :b :l",
                            d_a, n, d_out, SCL_DEFAULT_WG_SIZE * sizeof(*h_a));

    /* Copy d_out from device memory to host memory */
    sclMemcpyDeviceToHost(h_out, d_out, SIZE_OUT);

    /* Perform the final reduction on the CPU */
    int s = 0;
    for (int i=0; i<N_OF_GROUPS; i++) {
        s += h_out[i];
    }
    /* Check result */
    const int expected = 2*n;
    if ( s != expected ) {
        printf("Check FAILED: got %d, expected %d\n", s, expected);
    } else {
        printf("Check OK: sum = %d\n", s);
    }
    /* Cleanup */
    free(h_a); free(h_out);
    sclFree(d_a); sclFree(d_out);
    sclFinalize();
    return EXIT_SUCCESS;
}
