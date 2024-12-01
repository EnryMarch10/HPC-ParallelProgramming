/****************************************************************************
 *
 * cuda-anneal.cu - ANNEAL cellular automaton
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

/* The following function simplifies indexing of the 2D
   domain. Instead of writing grid[i*ext_width + j] you write
   IDX(grid, ext_width, i, j) to get a pointer to grid[i][j]. This
   function assumes that the size of the CA grid is
   (ext_width*ext_height), where the first and last rows/columns are
   ghost cells. */
cell_t* IDX(cell_t *grid, int ext_width, int i, int j)
{
    return (grid + i * ext_width + j);
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
void copy_top_bottom(cell_t *grid, int ext_width, int ext_height)
{
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    const int TOP_GHOST = TOP - 1;
    const int BOTTOM_GHOST = BOTTOM + 1;

    for (int j = 0; j < ext_width; j++) {
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
void copy_left_right(cell_t *grid, int ext_width, int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;

    for (int i = 0; i < ext_height; i++) {
        *IDX(grid, ext_width, i, RIGHT_GHOST) = *IDX(grid, ext_width, i, LEFT); /* left column to right halo */
        *IDX(grid, ext_width, i, LEFT_GHOST) = *IDX(grid, ext_width, i, RIGHT); /* right column to left halo */
    }
}

/* Compute the `next` grid given the current configuration `cur`.
   Both grids have (ext_width * ext_height) elements.

   [TODO] This function should be transformed into a kernel. */
void step(cell_t *cur, cell_t *next, int ext_width, int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    for (int i = TOP; i <= BOTTOM; i++) {
        for (int j = LEFT; j <= RIGHT; j++) {
            int nblack = 0;
            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    nblack += *IDX(cur, ext_width, i + di, j + dj);
                }
            }
            *IDX(next, ext_width, i, j) = (nblack >= 6 || nblack == 4);
        }
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
    cell_t *cur, *next;
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
    next = (cell_t *) malloc(ext_size);
    assert(next != NULL);
    init(cur, ext_width, ext_height, 0.5);
    const double tstart = hpc_gettime();
    for (s = 0; s < nsteps; s++) {
        copy_top_bottom(cur, ext_width, ext_height);
        copy_left_right(cur, ext_width, ext_height);
#ifdef DUMPALL
        write_pbm(cur, ext_width, ext_height, s);
#endif
        step(cur, next, ext_width, ext_height);
        cell_t *tmp = cur;
        cur = next;
        next = tmp;
    }
    const double elapsed = hpc_gettime() - tstart;
    write_pbm(cur, ext_width, ext_height, s);
    free(cur);
    free(next);
    fprintf(stderr, "Elapsed time: %f (%f Mops/s)\n", elapsed, (width * height / 1.0e6) * nsteps / elapsed);

    return EXIT_SUCCESS;
}
