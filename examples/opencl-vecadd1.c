/****************************************************************************
 *
 * opencl-vecadd1.c - Sum two arrays with OpenCL, using work groups
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
 * Compile with:
 * cc opencl-vecadd1.c simpleCL.c -o opencl-vecadd1 -lOpenCL
 *
 * Run with:
 * ./opencl-vecadd3
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "simpleCL.h"

const char *source =
    "__kernel void add_kernel( __global const int *a,\n"
    "                          __global const int *b,\n"
    "                          __global int *c,\n"
    "                          int n )\n"
    "{\n"
    "    const int i = get_global_id(0);\n"
    "    if ( i < n ) {\n"
    "        c[i] = a[i] + b[i];\n"
    "    }\n"
    "}\n";

void vec_init( int *a, int n )
{
    int i;
    for (i=0; i<n; i++) {
        a[i] = i;
    }
}

int main( void )
{
    sclInitFromString(source);

    int *a, *b, *c;             /* host copies of a, b, c */
    cl_mem d_a, d_b, d_c;	/* device copies of a, b, c */
    int i;
    const size_t N = 1024*1024;
    const size_t SIZE = N*sizeof(int);

    /* Allocate space for host copies of a, b, c */
    a = (int*)malloc(SIZE); vec_init(a, N);
    b = (int*)malloc(SIZE); vec_init(b, N);
    c = (int*)malloc(SIZE);

    /* Allocate space for device copies of a, b, c */
    d_a = sclMallocCopy(SIZE, a, CL_MEM_READ_ONLY);
    d_b = sclMallocCopy(SIZE, b, CL_MEM_READ_ONLY);
    d_c = sclMalloc(SIZE, CL_MEM_WRITE_ONLY);

    /* Launch add() kernel on GPU */
    const sclDim GRID = DIM1(sclRoundUp(N, SCL_DEFAULT_WG_SIZE));
    const sclDim BLOCK = DIM1(SCL_DEFAULT_WG_SIZE);
    sclSetArgsEnqueueKernel(sclCreateKernel("add_kernel"),
                            GRID, BLOCK,
                            ":b :b :b :d",
                            d_a, d_b, d_c, N);

    /* Copy result back to host */
    sclMemcpyDeviceToHost(c, d_c, SIZE);

    /* Check results */
    for (i=0; i<N; i++) {
        if ( c[i] != a[i] + b[i] ) {
            fprintf(stderr, "Error at index %d: a[%d]=%d, b[%d]=%d, c[%d]=%d\n",
                    i, i, a[i], i, b[i], i, c[i]);
            break;
        }
    }
    if (i == N) {
        printf("Check OK\n");
    }
    /* Cleanup */
    free(a); free(b); free(c);
    sclFree(d_a); sclFree(d_b); sclFree(d_c);
    sclFinalize();
    return EXIT_SUCCESS;
}
