/****************************************************************************
 *
 * cuda-multigpu.cu - CUDA multi GPU example
 *
 * Copyright (C) 2024 Moreno Marzolla
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

#include <stdio.h>
#include <assert.h>

#define BLKDIM 1024

/* compute the sum of the content of v[BLKDIM] and store the result in v[0] */
__global__ void reduce(int *v)
{
    __shared__ int temp[BLKDIM];
    int lindex = threadIdx.x;
    int bsize = blockDim.x / 2;

    temp[lindex] = v[lindex];

    /* wait for all threads to finish the copy operation */
    __syncthreads();

    /* All threads within the block cooperate to compute the local sum */
    while (bsize > 0) {
        if (lindex < bsize) {
            temp[lindex] += temp[lindex + bsize];
        }
        bsize = bsize / 2;
        /* threads must synchronize before performing the next
           reduction step */
        __syncthreads();
    }

    if (lindex == 0) {
        v[0] = temp[0];
    }
}

void init(int *v, int len, int val)
{
    for (int i = 0; i < len; i++) {
        v[i] = val;
    }
}

int main(void)
{
    int *p0, *d_p0, *p1, *d_p1;
    int r0, r1;

    const size_t size = BLKDIM * sizeof(*p0);

    cudaMallocHost(&p0, size);
    init(p0, BLKDIM, 1);
    cudaMallocHost(&p1, size);
    init(p1, BLKDIM, 2);

    cudaSetDevice(0);            // Set device 0 as current
    cudaMalloc(&d_p0, size);     // Allocate memory on device 0
    cudaMemcpyAsync(d_p0, p0, size, cudaMemcpyHostToDevice);
    reduce<<<1, BLKDIM>>>(d_p0); // Launch kernel on device 0
    cudaMemcpyAsync(&r0, d_p0, sizeof(r0), cudaMemcpyDeviceToHost);

    cudaSetDevice(1);            // Set device 1 as current
    cudaMalloc(&d_p1, size);     // Allocate memory on device 1
    cudaMemcpyAsync(d_p1, p1, size, cudaMemcpyHostToDevice);
    reduce<<<1, BLKDIM>>>(d_p1); // Launch kernel on device 1
    cudaMemcpyAsync(&r1, d_p1, sizeof(r1), cudaMemcpyDeviceToHost);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaFree(d_p0);
    // the CPU can use r0 here
    printf("r0 = %d\n", r0);
    assert(r0 == BLKDIM);

    cudaSetDevice(1);
    cudaDeviceSynchronize();
    cudaFree(d_p1);
    // the CPU can use r1 here
    printf("r1 = %d\n", r1);
    assert(r1 == 2 * BLKDIM);

    cudaFreeHost(p0);
    cudaFreeHost(p1);

    return 0;
}
