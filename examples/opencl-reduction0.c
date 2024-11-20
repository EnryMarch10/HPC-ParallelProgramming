/****************************************************************************
 *
 * opencl-reduction0.c - Reduction with OpenCL
 *
 * Basic version: only the first thread of each block does a partial
 * reduction.
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
 * This program realizes a simple sum-reduction on the GPU. This
 * implementation is not efficient: each workgroup copies a portion of
 * the array in local memory; work-item 0 of each workgroup computes a
 * partial sum of the local data. The final reduction must be
 * completed by the CPU.
 *
 * Compile with:
 * cc opencl-reduction0.c simpleCL.c -o opencl-reduction0 -lOpenCL
 *
 * Run with:
 * ./opencl-reduction0
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

int main(void)
{
    sclInitFromFile("opencl-reduction0.cl");

    const int N_OF_BLOCKS = 1024;
    const int N = N_OF_BLOCKS * SCL_DEFAULT_WG_SIZE;
    int *h_a; int h_sums[N_OF_BLOCKS];
    cl_mem d_a, d_sums;
    int i, s = 0;

    assert(N % SCL_DEFAULT_WG_SIZE == 0);
    /* Allocate space for host copy of the array */
    h_a = (int *) malloc(N * sizeof(int));
    /* Initialize h_a[] deterministically, so that we know that the
       result of the sum must be 2*N */
    for (i = 0; i < N; i++) { h_a[i] = 2; }
    /* Allocate space for device copies of d_a and d_sums */
    d_a = sclMallocCopy(N * sizeof(*h_a), h_a, CL_MEM_READ_ONLY);
    d_sums = sclMalloc(sizeof(h_sums), CL_MEM_WRITE_ONLY);
    /* Launch the reduction kernel on the GPU */
    sclSetArgsLaunchKernel(sclCreateKernel("reduction_kernel"),
                           DIM1(N), DIM1(SCL_DEFAULT_WG_SIZE),
                           ":b :b :d :l",
                           d_a, d_sums, N, SCL_DEFAULT_WG_SIZE * sizeof(*h_a));
    /* Copy the d_sums[] array from device memory to host memory h_sums[] */
    sclMemcpyDeviceToHost(h_sums, d_sums, N_OF_BLOCKS * sizeof(*h_sums));
    /* Perform the final reduction on the CPU */
    s = 0;
    for (i = 0; i < N_OF_BLOCKS; i++) {
        s += h_sums[i];
    }
    /* Check result */
    if (s != 2 * N) {
        printf("Check FAILED: Expected %d, got %d\n", 2 * N, s);
    } else {
        printf("Check OK: sum = %d\n", s);
    }
    /* Cleanup */
    free(h_a);
    sclFree(d_a);
    sclFree(d_sums);
    sclFinalize();
    return EXIT_SUCCESS;
}
