/****************************************************************************
 *
 * opencl-reduction2.c - Reduction with OpenCL atomic operations
 *
 * This version works for any array length n; however, it still
 * requires that BLKDIM is a power of two.
 *
 * Copyright (C) 2019 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * GPU. Each thread block copies a portion of the array in shared
 * memory; then, all threads within the same block cooperate to
 * compute the sum of the local data by organizing the computation as
 * a tree. Atomic operations are used to complete the execution
 * using the GPU only.
 *
 * Compile with:
 * cc opencl-reduction3.c simpleCL.c -o opencl-reduction3 -lOpenCL
 *
 * Run with:
 * ./opencl-reduction3
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

void init(int *v, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        v[i] = 2;
    }
}

int main(int argc, char *argv[])
{
    sclInitFromFile("opencl-reduction3.cl");

    int *h_a, result = 0;
    cl_mem d_a, d_result;
    int n = 8192;

    assert((SCL_DEFAULT_WG_SIZE & (SCL_DEFAULT_WG_SIZE-1)) == 0); /* check if the default group size is a power of two using the "bit hack" from http://www.graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2 */

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    const size_t SIZE = n * sizeof(*h_a);
    const int N_OF_GROUPS = (n + SCL_DEFAULT_WG_SIZE - 1) / SCL_DEFAULT_WG_SIZE;

    /* Allocate space for host copies of array */
    h_a = (int *) malloc(SIZE);
    assert(h_a != NULL);
    init(h_a, n);

    /* Allocate space for device copies of aarray */
    d_a = sclMallocCopy(SIZE, h_a, CL_MEM_READ_ONLY);
    d_result = sclMallocCopy(sizeof(result), &result, CL_MEM_READ_WRITE);

    /* Launch sum() kernel on the GPU */
    sclSetArgsEnqueueKernel(sclCreateKernel("sum_kernel"),
                            DIM1(N_OF_GROUPS * SCL_DEFAULT_WG_SIZE), DIM1(SCL_DEFAULT_WG_SIZE),
                            ":b :d :b :l",
                            d_a, n, d_result, SCL_DEFAULT_WG_SIZE*sizeof(*h_a));

    /* Copy the result from device memory to host memory */
    sclMemcpyDeviceToHost(&result, d_result, sizeof(result));

    /* Check result */
    const int expected = 2 * n;
    if (result != expected) {
        printf("Check FAILED: got %d, expected %d\n", result, expected);
    } else {
        printf("Check OK: sum = %d\n", result);
    }
    /* Cleanup */
    free(h_a);
    sclFree(d_a);
    sclFree(d_result);
    sclFinalize();
    return EXIT_SUCCESS;
}
