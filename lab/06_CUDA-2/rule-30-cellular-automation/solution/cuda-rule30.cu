/****************************************************************************
 *
 * cuda-rule30.cu - "Rule 30" Callular Automaton
 *
 * Copyright (C) 2017--2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

typedef unsigned char cell_t;

#define BLKDIM 1024

__device__ int d_min(int a, int b)
{
    return a < b ? a : b;
}

/**
 * Fill ghost cells in device memory. This kernel must be launched
 * with one thread only.
 */
__global__ void fill_ghost(cell_t *cur, int ext_n)
{
    const int LEFT_GHOST = 0;
    const int LEFT = 1;
    const int RIGHT_GHOST = ext_n - 1;
    const int RIGHT = RIGHT_GHOST - 1;
    cur[RIGHT_GHOST] = cur[LEFT];
    cur[LEFT_GHOST] = cur[RIGHT];
}

/**
 * Given the current state `cur` of the CA, compute the `next`
 * state. This function requires that `cur` and `next` are extended
 * with ghost cells; therefore, `ext_n` is the lenght of `cur` and
 * `next` _including_ ghost cells.
 */
__global__ void step(cell_t *cur, cell_t *next, int ext_n)
{
    __shared__ cell_t buf[BLKDIM + 2];
    const int gindex = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    const int lindex = 1 + threadIdx.x;

    if (gindex < ext_n - 1) {
        buf[lindex] = cur[gindex];
        if (lindex == 1) {
            /* The thread with threadIdx.x == 0 (therefore, with
               lindex == 1) fills the two ghost cells of `buf[]` (one
               on the left, one on the right). When the width of the
               domain (ext_n - 2) is not multiple of BLKDIM, care must
               be taken. Indeed, if the width is not multiple of
               `BLKDIM`, then the rightmost ghost cell of the last
               thread block is `buf[1+len]`, where len is computed as
               follows: */
            const int len = d_min(BLKDIM, ext_n - 1 - gindex);
            buf[0] = cur[gindex - 1];
            buf[1 + len] = cur[gindex + len];
        }

        __syncthreads();

        const cell_t left   = buf[lindex - 1];
        const cell_t center = buf[lindex];
        const cell_t right  = buf[lindex + 1];

        next[gindex] =
            ( left && !center && !right) ||
            (!left && !center &&  right) ||
            (!left &&  center && !right) ||
            (!left &&  center &&  right);
    }
}

/**
 * Initialize the domain; all cells are 0, with the exception of a
 * single cell in the middle of the domain. `cur` points to an array
 * of length `ext_n`; the length includes two ghost cells.
 */
void init_domain(cell_t *cur, int ext_n)
{
    for (int i = 0; i < ext_n; i++) {
        cur[i] = 0;
    }
    cur[ext_n / 2] = 1;
}

/**
 * Dump the current state of the CA to PBM file `out`. `cur` points to
 * an array of length `ext_n` that includes two ghost cells.
 */
void dump_state(FILE *out, const cell_t *cur, int ext_n)
{
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    for (int i = LEFT; i <= RIGHT; i++) {
        fprintf(out, "%d ", cur[i]);
    }
    fprintf(out, "\n");
}

int main(int argc, char *argv[])
{
    const char *outname = "cuda-rule30.pbm";
    FILE *out;
    int width = 1024, steps = 1024;
    cell_t *cur;
    cell_t *d_cur, *d_next;

    if (argc > 3) {
        fprintf(stderr, "Usage: %s [width [steps]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        width = atoi(argv[1]);
    }

    if (argc > 2) {
        steps = atoi(argv[2]);
    }

    const int ext_width = width + 2;
    const size_t ext_size = ext_width * sizeof(*cur); /* includes ghost cells */
    /* Create the output file */
    out = fopen(outname, "w");
    if (!out) {
        fprintf(stderr, "FATAL: cannot create file \"%s\"\n", outname);
        return EXIT_FAILURE;
    }
    fprintf(out, "P1\n");
    fprintf(out, "# produced by cuda-rule30.cu\n");
    fprintf(out, "%d %d\n", width, steps);

    /* Allocate space for `d_cur[]` and `d_next[]` on the device */
    cudaSafeCall(cudaMalloc((void **) &d_cur, ext_size));
    cudaSafeCall(cudaMalloc((void **) &d_next, ext_size));

    /* Allocate space for host copy of `cur[]` */
    cur = (cell_t *) malloc(ext_size);
    assert(cur != NULL);

    /* Initialize the domain */
    init_domain(cur, ext_width);

    /* Copy input to device */
    cudaSafeCall(cudaMemcpy(d_cur, cur, ext_size, cudaMemcpyHostToDevice));

    /* Evolve the CA */
    for (int s = 0; s < steps; s++) {
        /* Dump the current state to the output image */
        dump_state(out, cur, ext_width);

        /* Fill ghost cells */
        fill_ghost<<<1, 1>>>(d_cur, ext_width);
        cudaCheckError();

        /* Compute next state */
        step<<<(width + BLKDIM - 1) / BLKDIM, BLKDIM>>>(d_cur, d_next, ext_width);
        cudaCheckError();

        cudaSafeCall(cudaMemcpy(cur, d_next, ext_size, cudaMemcpyDeviceToHost));

        /* swap d_cur and d_next */
        cell_t *d_tmp = d_cur;
        d_cur = d_next;
        d_next = d_tmp;
    }

    free(cur);
    cudaFree(d_cur);
    cudaFree(d_next);

    fclose(out);

    return EXIT_SUCCESS;
}
