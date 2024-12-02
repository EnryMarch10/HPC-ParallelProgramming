#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

#define BLKDIM 1024

typedef unsigned char cell_t;

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
 * Given the current state of the CA, compute the next state.  This
 * version requires that the `cur` and `next` arrays are extended with
 * ghost cells; therefore, `ext_n` is the length of `cur` and `next`
 * _including_ ghost cells.
 *
 *                             +----- ext_n-2
 *                             |   +- ext_n-1
 *   0   1                     V   V
 * +---+-------------------------+---+
 * |///|                         |///|
 * +---+-------------------------+---+
 *
 */
__global__ void rule30_step(cell_t *cur, cell_t *next, int n, int ext_n) {
    __shared__ int cache[BLKDIM + 2];

    const int l_idx = 1 + threadIdx.x;
    const int g_idx = 1 + threadIdx.x + blockIdx.x * blockDim.x;

    const int remaining = n - blockIdx.x * blockDim.x;
    const int my_dim = remaining >= BLKDIM ? BLKDIM : remaining;

    if (l_idx == 1) { /* same as if I am first thread of block */
        cache[0] = cur[g_idx - 1];
        cache[my_dim + 1] = cur[g_idx + my_dim];
    }
    if (g_idx < ext_n - 1) {
        cache[l_idx] = cur[g_idx];
        __syncthreads();
        /* I decided not to save the following variables in local variables,
           because these could be registers (better option) or global memory
           (worst option). It should be checked which option is better.
         */
        next[g_idx] =
            ( cache[l_idx - 1] && !cache[l_idx] && !cache[l_idx + 1]) ||
            (!cache[l_idx - 1] && !cache[l_idx] &&  cache[l_idx + 1]) ||
            (!cache[l_idx - 1] &&  cache[l_idx] && !cache[l_idx + 1]) ||
            (!cache[l_idx - 1] &&  cache[l_idx] &&  cache[l_idx + 1]);
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

    /* Allocate space for the `cur[]` and `next[]` arrays */
    cur = (cell_t *) malloc(ext_size);
    assert(cur != NULL);

    /* Initialize the domain */
    init_domain(cur, ext_width);

    cell_t *d_cur, *d_next;

    cudaSafeCall(cudaMalloc((void **) &d_cur, ext_width));
    cudaSafeCall(cudaMalloc((void **) &d_next, ext_width));

    cudaSafeCall(cudaMemcpy(d_cur, cur, ext_width, cudaMemcpyHostToDevice));

    const int GRID = (width + BLKDIM - 1) / BLKDIM;

    /* Evolve the CA */
    for (int s = 0; s < steps; s++) {
        /* Dump the current state */
        dump_state(out, cur, ext_width);

        fill_ghost<<<1, 1>>>(d_cur, ext_width);
        cudaCheckError();
    
        /* Compute next state */
        rule30_step<<<GRID, BLKDIM>>>(d_cur, d_next, width, ext_width);
        cudaCheckError();

        cudaSafeCall(cudaMemcpy(cur, d_next, ext_width, cudaMemcpyDeviceToHost));

        /* swap d_cur and d_next */
        cell_t *d_tmp = d_cur;
        d_cur = d_next;
        d_next = d_tmp;
    }

    cudaSafeCall(cudaFree(d_cur));
    cudaSafeCall(cudaFree(d_next));

    free(cur);

    fclose(out);

    return EXIT_SUCCESS;
}
