/****************************************************************************
 *
 * opencl-vecadd0.c - Sum two integers with OpenCL
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
 * cc opencl-vecadd0.c simpleCL.c -o opencl-vecadd0 -lOpenCL
 *
 * Run with:
 * ./opencl-vecadd0
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "simpleCL.h"

const char* source =
    "__kernel void add_kernel( __global const int *a,\n"
    "                          __global const int *b,\n"
    "                          __global int *c )\n"
    "{\n"
    "    *c = *a + *b;\n"
    "}\n";

int main( void )
{
    int a, b, c;	          /* host copies of a, b, c */
    cl_mem d_a, d_b, d_c;	  /* device copies of a, b, c */
    const size_t size = sizeof(int);
    sclInitFromString(source);
    /* Setup input values */
    a = 2; b = 7;
    /* Allocate space for device copies of a, b, c */
    d_a = sclMallocCopy(size, &a, CL_MEM_READ_ONLY);
    d_b = sclMallocCopy(size, &b, CL_MEM_READ_ONLY);
    d_c = sclMalloc(size, CL_MEM_WRITE_ONLY);
    /* Launch add() kernel on GPU */
    sclSetArgsEnqueueKernel(sclCreateKernel("add_kernel"),
                            DIM1(1), DIM1(1),
                            ":b :b :b",
                            d_a, d_b, d_c);
    /* Copy result back to host */
    sclMemcpyDeviceToHost(&c, d_c, size);
    /* check result */
    if ( c != a + b ) {
        fprintf(stderr, "Test FAILED: expected %d, got %d\n", a+b, c);
    } else {
        printf("Test OK\n");
    }
    /* Cleanup */
    sclFree(d_a); sclFree(d_b); sclFree(d_c);
    sclFinalize();
    return EXIT_SUCCESS;
}
