#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

typedef unsigned char cell_t;

/* The following function simplifies indexing of the 2D
   domain. Instead of writing grid[i*ext_width + j] you write
   IDX(grid, ext_width, i, j) to get a pointer to grid[i][j]. This
   function assumes that the size of the CA grid is
   (ext_width*ext_height), where the first and last rows/columns are
   ghost cells. */
__host__ __device__ cell_t* IDX(cell_t *grid, int ext_width, int i, int j)
{
    return grid + i * ext_width + j;
}

/*
  `grid` points to a (ext_width * ext_height) block of bytes; this
  function copies the top and bottom ext_width elements to the
  opposite halo (see figure below).

   LEFT_GHOST=0     RIGHT=ext_width-2
   | LEFT=1         | RIGHT_GHOST=ext_width-1
   | |              | |
   v v              v v
  +-+----------------+-+
  |Y|YYYYYYYYYYYYYYYY|Y| <- TOP_GHOST=0
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- TOP=1
  | |                | |
  | |                | |
  | |                | |
  | |                | |
  |Y|YYYYYYYYYYYYYYYY|Y| <- BOTTOM=ext_height - 2
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- BOTTOM_GHOST=ext_height - 1
  +-+----------------+-+

 */
/* [TODO] Transform this function into a kernel */
__global__ void copy_top_bottom(cell_t *grid, int ext_width, int ext_height)
{
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    const int TOP_GHOST = TOP - 1;
    const int BOTTOM_GHOST = BOTTOM + 1;
    const int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < ext_width) {
        *IDX(grid, ext_width, BOTTOM_GHOST, j) = *IDX(grid, ext_width, TOP, j); /* top to bottom halo */
        *IDX(grid, ext_width, TOP_GHOST, j) = *IDX(grid, ext_width, BOTTOM, j); /* bottom to top halo */
    }
}

/*
  `grid` points to a ext_width*ext_height block of bytes; this
  function copies the left and right ext_height elements to the
  opposite halo (see figure below).

   LEFT_GHOST=0     RIGHT=ext_width-2
   | LEFT=1         | RIGHT_GHOST=ext_width-1
   | |              | |
   v v              v v
  +-+----------------+-+
  |X|Y              X|Y| <- TOP_GHOST=0
  +-+----------------+-+
  |X|Y              X|Y| <- TOP=1
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y| <- BOTTOM=ext_height - 2
  +-+----------------+-+
  |X|Y              X|Y| <- BOTTOM_GHOST=ext_height - 1
  +-+----------------+-+

 */
/* [TODO] This function should be transformed into a kernel */
__global__ void copy_left_right(cell_t *grid, int ext_width, int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < ext_height) {
        *IDX(grid, ext_width, i, RIGHT_GHOST) = *IDX(grid, ext_width, i, LEFT); /* left column to right halo */
        *IDX(grid, ext_width, i, LEFT_GHOST) = *IDX(grid, ext_width, i, RIGHT); /* right column to left halo */
    }
}

/* Compute the `next` grid given the current configuration `cur`.
   Both grids have (ext_width * ext_height) elements.

   [TODO] This function should be transformed into a kernel. */
__global__ void step(cell_t *cur, cell_t *next, int ext_width, int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    const int i = threadIdx.x + blockIdx.x * blockDim.x + TOP; /* We do not want to consider the ghost area */
    const int j = threadIdx.y + blockIdx.y * blockDim.y + LEFT; /* We do not want to consider the ghost area */

    if (i <= BOTTOM && j <= RIGHT) {
        int nblack = 0;
#pragma unroll /* It is a LOOP UNROLLING technique, it is used by compilers by default => code longer but more efficient */
        for (int di = -1; di <= 1; di++) {
#pragma unroll
            for (int dj = -1; dj <= 1; dj++) {
                nblack += *IDX(cur, ext_width, i + di, j + dj);
            }
        }
        *IDX(next, ext_width, i, j) = (nblack >= 6 || nblack == 4);
    }
}

/* Initialize the current grid `cur` with alive cells with density `p`. */
void init(cell_t *cur, int ext_width, int ext_height, float p)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;

    srand(1234); /* initialize PRND */
    for (int i = TOP; i <= BOTTOM; i++) {
        for (int j = LEFT; j <= RIGHT; j++) {
            *IDX(cur, ext_width, i, j) = (((float) rand()) / RAND_MAX < p);
        }
    }
}

/* Write `cur` to a PBM (Portable Bitmap) file whose name is derived from the step number `stepno`. */
void write_pbm(cell_t *cur, int ext_width, int ext_height, int stepno)
{
    char fname[128];
    FILE *f;
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;

    snprintf(fname, sizeof(fname), "cuda-anneal-%06d.pbm", stepno);

    if ((f = fopen(fname, "w")) == NULL) {
        fprintf(stderr, "Cannot open %s for writing\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(f, "P1\n");
    fprintf(f, "# produced by cuda-anneal.cu\n");
    fprintf(f, "%d %d\n", ext_width - 2, ext_height - 2);
    for (int i = LEFT; i <= RIGHT; i++) {
        for (int j = TOP; j <= BOTTOM; j++) {
            fprintf(f, "%d ", *IDX(cur, ext_width, i, j));
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main(int argc, char *argv[])
{
    cell_t *cur;
    cell_t *d_cur, *d_next;
    int nsteps = 64, width = 512, height = 512, s;
    const int MAXN = 2048;

    if (argc > 4) {
        fprintf(stderr, "Usage: %s [nsteps [W [H]]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1) {
        nsteps = atoi(argv[1]);
    }

    if (argc > 2) {
        width = height = atoi(argv[2]);
    }

    if (argc > 3) {
        height = atoi(argv[3]);
    }

    if (width > MAXN || height > MAXN) { /* maximum image size is MAXN */
        fprintf(stderr, "FATAL: the maximum allowed grid size is %d\n", MAXN);
        return EXIT_FAILURE;
    }

    const int ext_width = width + 2;
    const int ext_height = height + 2;
    const size_t ext_size = ext_width * ext_height * sizeof(cell_t);

    fprintf(stderr, "Anneal CA: steps=%d size=%d x %d\n", nsteps, width, height);

    cur = (cell_t *) malloc(ext_size);
    assert(cur != NULL);
    cudaSafeCall(cudaMalloc((void **) &d_cur, ext_size));
    cudaSafeCall(cudaMalloc((void **) &d_next, ext_size));

    init(cur, ext_width, ext_height, 0.5);
    cudaSafeCall(cudaMemcpy(d_cur, cur, ext_size, cudaMemcpyHostToDevice));

    const int BLKDIM = 1024;
    const dim3 copyTBBlock(BLKDIM);
    const dim3 copyTBGrid((ext_width + BLKDIM - 1) / BLKDIM);
    const dim3 copyLRBlock(BLKDIM);
    const dim3 copyLRGrid((ext_height + BLKDIM - 1) / BLKDIM);

    const int BLKDIM_STEP = 32;
    const dim3 stepBlock(BLKDIM_STEP, BLKDIM_STEP);
    const dim3 stepGrid((width + BLKDIM_STEP - 1) / BLKDIM_STEP, (height + BLKDIM_STEP - 1) / BLKDIM_STEP);

    const double tstart = hpc_gettime();
    for (s = 0; s < nsteps; s++) {
        copy_top_bottom<<<copyTBGrid, copyTBBlock>>>(d_cur, ext_width, ext_height);
        cudaCheckError();
        copy_left_right<<<copyLRGrid, copyLRBlock>>>(d_cur, ext_width, ext_height);
        cudaCheckError();
#ifdef DUMPALL
        cudaSafeCall(cudaMemcpy(cur, d_cur, ext_size, cudaMemcpyDeviceToHost));
        write_pbm(cur, ext_width, ext_height, s);
#endif
        step<<<stepGrid, stepBlock>>>(d_cur, d_next, ext_width, ext_height);
        cudaCheckError();
        cell_t *tmp = d_cur;
        d_cur = d_next;
        d_next = tmp;
    }
    cudaSafeCall(cudaDeviceSynchronize());
    const double elapsed = hpc_gettime() - tstart;
    cudaSafeCall(cudaMemcpy(cur, d_cur, ext_size, cudaMemcpyDeviceToHost));
    write_pbm(cur, ext_width, ext_height, s);
    cudaSafeCall(cudaFree(d_cur));
    free(cur);
    fprintf(stderr, "Elapsed time: %f (%f Mops/s)\n", elapsed, (width * height / 1.0e6) * nsteps / elapsed);

    return EXIT_SUCCESS;
}
