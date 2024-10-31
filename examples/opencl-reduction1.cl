/****************************************************************************
 *
 * opencl-reduction1.cl -- Kernel for opencl-reduction1.c
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

/* This kernel copies a portion of the array a[] of length n into
   local memory. All work-items cooperate to compute the sum of the
   local data; at the end, thread 0 stores the computed value on the
   appropriate entry of d_sums[]. Different workgroups access
   different elements of d_sums[], so no race condition is possible. */
__kernel void
sum_kernel( __global const int *a,
            int n,
            __global int *d_sums,
            __local int *temp )
{
    const int lindex = get_local_id(0);
    const int bindex = get_group_id(0);
    const int gindex = get_global_id(0);

    temp[lindex] = a[gindex];
    /* wait for all work-items to finish the copy operation */

    barrier(CLK_LOCAL_MEM_FENCE);
    /* All work-items cooperate to compute the local sum */

    for ( int bsize = get_local_size(0) / 2; bsize > 0; bsize /= 2 ) {
        if ( lindex < bsize ) {
            temp[lindex] += temp[lindex + bsize];
        }
        /* work-items must synchronize before performing the next
           reduction step */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if ( 0 == lindex ) {
        /* Work-item 0 copies the local sum into the appropriate
           element of d_sums[] */
        d_sums[bindex] = temp[0];
    }
}
