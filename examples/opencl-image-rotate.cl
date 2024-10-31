/****************************************************************************
 *
 * opencl-image-rotate.cl -- Kernels for opencl-image-rotate.c
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

/**
 * Rotate square image `orig` of size `n`x`n` 90 degrees clockwise;
 * the new image goes to `rotated`. This kernel works for any image
 * size.
 */
__kernel void
rotate_kernel( __global const unsigned char *orig,
               __global unsigned char *rotated,
               int n )
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x<n && y<n) {
        rotated[n*x + (n - 1 - y)] = orig[n*y + x];
    }
}

/**
 * Same as above, using shared memory. This kernel requires that `n`
 * is a multiple of the workgroup size.
 */
__kernel void
rotate_kernel_shared( __global const unsigned char *orig,
                      __global unsigned char *rotated,
                      int n )
{
    __local unsigned char buf[SCL_DEFAULT_WG_SIZE2D][SCL_DEFAULT_WG_SIZE2D];
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int BSIZE = get_local_size(0);

    if (x<n && y<n) {
        const int block_x = get_group_id(0);
        const int block_y = get_group_id(1);
        const int local_x = get_local_id(0);
        const int local_y = get_local_id(1);
        const int nblocks_x = get_num_groups(0);
        const int nblocks_y = get_num_groups(1);

        buf[local_x][BSIZE-1-local_y] = orig[n*y+x];
        barrier(CLK_LOCAL_MEM_FENCE);

        const int dest_x = (nblocks_y - 1 - block_y) * BSIZE + local_x;
        const int dest_y = block_x * BSIZE + local_y;
        rotated[n*dest_y + dest_x] = buf[local_y][local_x];
    }
}
