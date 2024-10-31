/****************************************************************************
 *
 * opencl-stencil1d.c - 1D stencil example with OpenCL using local memory
 *
 * Copyright (C) 2021 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * This implementation does not use local memory
 *
 * Compile with:
 * cc opencl-stencil1d-local.c simpleCL.c -o opencl-stencil1d-local -lOpenCL
 *
 * Run with:
 * ./opencl-stencil1d-local
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

int main( void )
{
    sclInitFromFile("opencl-stencil1d.cl");

    int *h_in, *h_out;	  /* host copies of in and out */
    cl_mem d_in, d_out;	  /* device copies of in and out */
    int i;
    const int RADIUS = 3;
    const size_t N = SCL_DEFAULT_WG_SIZE * 1024; /* Input length EXCLUDING the first and last RADIUS elements */
    const size_t SIZE = (N+2*RADIUS)*sizeof(*h_in); /* input size */

    assert( N % SCL_DEFAULT_WG_SIZE == 0 );

    /* Allocate space for host copies of h_in and h_out */
    h_in = (int*)malloc(SIZE); assert(h_in != NULL);
    h_out = (int*)malloc(SIZE); assert(h_out != NULL);
    /* Initialize h_in[] */
    for (i=0; i<N+2*RADIUS; i++) { h_in[i] = 1; }

    /* Allocate space for device copies of d_in and d_out */
    d_in = sclMallocCopy(SIZE, h_in, CL_MEM_READ_ONLY);
    d_out = sclMalloc(SIZE, CL_MEM_WRITE_ONLY);
    /* Launch the kernel */
    sclSetArgsEnqueueKernel(sclCreateKernel("stencil1d_local"),
                            DIM1(N), DIM1(SCL_DEFAULT_WG_SIZE),
                            ":b :b :d :l",
                            d_in, d_out, RADIUS, (SCL_DEFAULT_WG_SIZE + 2*RADIUS) * sizeof(*h_in));
    /* Copy result back to host */
    sclMemcpyDeviceToHost(h_out, d_out, SIZE);
    /* Check the result */
    for (i=RADIUS; i<N+RADIUS; i++) {
        if ( h_out[i] != 7 ) {
            fprintf(stderr, "Error at index %d: h_out[%d] == %d, expected 7\n", i, i, h_out[i]);
            return EXIT_FAILURE;
        }
    }
    printf("Test OK\n");
    /* Cleanup */
    free(h_in); free(h_out);
    sclFree(d_in); sclFree(d_out);
    sclFinalize();
    return EXIT_SUCCESS;
}
