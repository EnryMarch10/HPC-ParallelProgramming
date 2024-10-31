/****************************************************************************
 *
 * opencl-stencil1d.cl -- Kernel for opencl-stencil1d.c
 *
 * Copyright 2021 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 ****************************************************************************/

/* This kernel does not use shared memoiry */
__kernel void
stencil1d(__global const int *in,
          __global int *out,
          int radius )
{
    const int i = get_global_id(0) + radius;
    int result = 0;
    for (int offset = -radius ; offset <= radius ; offset++) {
        result += in[i + offset];
    }
    /* Store the result */
    out[i] = result;
};

/* This kernel uses local memory.
   `local[]` must be an array of (bsize + 2*radius) elements */
__kernel void
stencil1d_local(__global const int *in,
                __global int *out,
                int radius,
                __local int *temp)
{
    const int gindex = get_global_id(0) + radius;
    const int lindex = get_local_id(0) + radius;
    const int bsize = get_local_size(0);
    int result = 0;
    /* Read input elements into local memory */
    temp[lindex] = in[gindex];
    if (get_local_id(0) < radius) {
        temp[lindex - radius] = in[gindex - radius];
        temp[lindex + bsize] = in[gindex + bsize];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    /* Apply the stencil */
    for (int offset = -radius ; offset <= radius ; offset++) {
        result += temp[lindex + offset];
    }
    /* Store the result */
    out[gindex] = result;
}
