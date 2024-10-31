/****************************************************************************
 *
 * opencl-reduction0.cl -- Kernel for opencl-reduction.
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

/* This kernel copies a portion of array a[] of n elements into
   thread-local shared memory. Thread 0 computes the sum of the local
   data, and stores the computed value on the appropriate entry of
   d_sums[]. Different thread blocks access different elements of
   d_sums[], so no race condition is possible.

   Local memory does not provide any performance advantage here, but
   is used to simplify extension of this kernel into more efficient
   ones. */
__kernel void
reduction_kernel( __global const int *a,
                  __global int *out,
                  int n,
                  __local int *temp)
{
    const int lindex = get_local_id(0);  /* index of the work-item within the workgroup */
    const int bindex = get_group_id(0);  /* index of the workgroup */
    const int gindex = get_global_id(0); /* index of the array element handled by this work-item */
    const int BSIZE = get_local_size(0); /* size of the workgroup */

    temp[lindex] = a[gindex];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* only work-item 0 computes the local sum */
    if ( 0 == lindex ) {
        int my_sum = 0;
        for (int i=0; i<BSIZE; i++) {
            my_sum += temp[i];
        }
        out[bindex] = my_sum;
    }
}
