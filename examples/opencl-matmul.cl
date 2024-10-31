/****************************************************************************
 *
 * opencl-matmul.cl -- Kernel for opencl-matmul.c
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

/* Compute r = p * q, for square nxn matrices p, q, r; this version
   does not use shared memory. */
__kernel void
matmul( __global const float *p,
        __global const float *q,
        __global float *r,
        int n )
{
    const int i = get_global_id(1);
    const int j = get_global_id(0);
    int k;
    float val = 0.0;
    if ( i < n && j < n ) {
        for (k=0; k<n; k++) {
            val += p[i*n + k] * q[k*n + j];
        }
        r[i*n + j] = val;
    }
}

/* Compute r = p * q, for square n x n matrices p, q, r; this version
   uses shared memory. This kernel requires that n is a multiple of
   SCL_DEFAULT_WG_SIZE2D */
__kernel void
matmulb( __global const float *p,
         __global const float *q,
         __global float *r,
         int n )
{
    __local float local_p[SCL_DEFAULT_WG_SIZE2D][SCL_DEFAULT_WG_SIZE2D];
    __local float local_q[SCL_DEFAULT_WG_SIZE2D][SCL_DEFAULT_WG_SIZE2D];
    const int bx = get_group_id(0), by = get_group_id(1);
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int BSIZE = get_local_size(0);
    const int i = by * BSIZE + ty;
    const int j = bx * BSIZE + tx;
    float v = 0.0;
    int m, k;
    for (m = 0; m < n; m += BSIZE) { /* loop over tiles */
        local_p[ty][tx] = p[i*n + (m + tx)];
        local_q[ty][tx] = q[(m + ty)*n + j];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (k = 0; k < BSIZE; k++) { /* loop within tile */
            v += local_p[ty][k] * local_q[k][tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    r[i*n + j] = v; /* write back to global memory */
}

/* Same as above, but does not require that n is a multiple of
   BSIZE. To do so, it fills shared buffers so that values outside
   the matrices are treated as zeros. */
__kernel void
matmulb_generic( __global const float *p,
                 __global const float *q,
                 __global float *r,
                 int n )
{
    __local float local_p[SCL_DEFAULT_WG_SIZE2D][SCL_DEFAULT_WG_SIZE2D];
    __local float local_q[SCL_DEFAULT_WG_SIZE2D][SCL_DEFAULT_WG_SIZE2D];
    const int bx = get_group_id(0), by = get_group_id(1);
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int BSIZE = get_local_size(0);
    const int i = by * BSIZE + ty;
    const int j = bx * BSIZE + tx;
    float v = 0.0;
    for (int m = 0; m < n; m += BSIZE) { /* loop over tiles */
        local_p[ty][tx] = local_q[ty][tx] = 0;
        if (i<n && m+tx<n)
            local_p[ty][tx] = p[i*n + (m + tx)];
        if (j<n && m+ty<n)
            local_q[ty][tx] = q[(m + ty)*n + j];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < BSIZE; k++) { /* loop within tile */
            v += local_p[ty][k] * local_q[k][tx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i<n && j<n)
        r[i*n + j] = v; /* write result to global memory */
}
