/****************************************************************************
 *
 * opencl-reduction2.cl -- Kernel for opencl-reduction2.c
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

__kernel void
sum_kernel( __global const int *a,
            int n,
            __global int *out,
            __local int *temp )
{
    const int lindex = get_local_id(0);
    const int bindex = get_group_id(0);
    const int gindex = get_global_id(0);
    int bsize = get_local_size(0) / 2;

    if ( gindex < n ) {
        temp[lindex] = a[gindex];
    } else {
        /* Threads that are mapped outside the array a[] fill their
           corresponding entry of temp[] with 0 */
        temp[lindex] = 0;
    }

    /* wait for all work-items to finish the copy operation */
    barrier(CLK_LOCAL_MEM_FENCE);

    /* All work-items cooperate to compute the local sum */
    while ( bsize > 0 ) {
        if ( lindex < bsize ) {
            temp[lindex] += temp[lindex + bsize];
        }
        bsize = bsize / 2;
        /* threads must synchronize before performing the next
           reduction step */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if ( 0 == lindex ) {
        out[bindex] = temp[0];
    }
}
