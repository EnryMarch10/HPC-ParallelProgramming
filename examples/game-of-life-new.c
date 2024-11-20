/****************************************************************************
 *
 * game-of-life-new.c - Serial implementaiton of the Game of Life
 *
 * Copyright (C) 2017, 2021, 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * --------------------------------------------------------------------------
 *
 * Compile with:
 *
 *      gcc -std=c99 -Wall -Wpedantic game-of-life-new.c -o game-of-life-new
 *
 * Run with:
 *
 *      ./game-of-life-new 100
 *
 * To display the images
 *
 *      animate gol*.pbm
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

/* grid size (excluding ghost cells) */
#define SIZE 256

/* type of cell elements */
typedef unsigned char cell_t;

/* Simplifies indexing on a (SIZE+2)*(SIZE+2) grid */
int IDX(int i, int j)
{
    return i * (SIZE + 2) + j;
}

/* some useful constants; first and last rows/columns of the domain */
const int TOP    = 1;
const int BOTTOM = SIZE;
const int LEFT   = 1;
const int RIGHT  = SIZE;

/*
    LEFT          RIGHT
     |              |
     v              V
  +-+----------------+-+
  |\|\\\\\\\\\\\\\\\\|\|
  +-+----------------+-+
  |\|                |\| <- TOP
  |\|                |\|
  |\|                |\|
  |\|                |\|
  |\|                |\|
  |\|                |\| <- BOTTOM
  +-+----------------+-+
  |\|\\\\\\\\\\\\\\\\|\|
  +-+----------------+-+

 */

/* Fill the ghost cells of |grid| */
void copy_sides(cell_t *grid)
{
    int i, j;
    const int HALO_TOP    = TOP - 1;
    const int HALO_BOTTOM = BOTTOM + 1;
    const int HALO_LEFT   = LEFT - 1;
    const int HALO_RIGHT  = RIGHT + 1;

    /* copy top and bottom (one can also use memcpy() ) */
    for (j = LEFT; j < RIGHT + 1; j++) {
        grid[IDX(HALO_TOP,    j)] = grid[IDX(BOTTOM, j)];
        grid[IDX(HALO_BOTTOM, j)] = grid[IDX(TOP,    j)];
    }
    /* copy left and right */
    for (i = TOP; i < BOTTOM + 1; i++) {
        grid[IDX(i, HALO_LEFT )] = grid[IDX(i, RIGHT)];
        grid[IDX(i, HALO_RIGHT)] = grid[IDX(i, LEFT )];
    }
    /* copy corners */
    grid[IDX(HALO_TOP,    HALO_LEFT )] = grid[IDX(BOTTOM, RIGHT)];
    grid[IDX(HALO_TOP,    HALO_RIGHT)] = grid[IDX(BOTTOM, LEFT )];
    grid[IDX(HALO_BOTTOM, HALO_LEFT )] = grid[IDX(TOP,    RIGHT)];
    grid[IDX(HALO_BOTTOM, HALO_RIGHT)] = grid[IDX(TOP,    LEFT )];
}

/* Compute the |next| grid given the |cur|rent configuration. */
void step(cell_t *cur, cell_t *next)
{
    int i, j;
    for (i = TOP; i < BOTTOM + 1; i++) {
        for (j = LEFT; j < RIGHT + 1; j++) {
            /* count live neighbors of cell (i,j) */
            int nbors =
                cur[IDX(i - 1,j - 1)] + cur[IDX(i - 1,j)] + cur[IDX(i - 1,j + 1)] +
                cur[IDX(i    ,j - 1)] +                     cur[IDX(i    ,j + 1)] +
                cur[IDX(i + 1,j - 1)] + cur[IDX(i + 1,j)] + cur[IDX(i + 1,j + 1)];
 	    /* apply rules of the game of life to cell (i, j) */
            if (cur[IDX(i,j)] && (nbors < 2 || nbors > 3)) {
                next[IDX(i,j)] = 0;
            } else {
                if (!cur[IDX(i,j)] && (nbors == 3)) {
                    next[IDX(i,j)] = 1;
                } else {
                    next[IDX(i,j)] = cur[IDX(i,j)];
                }
            }
        }
    }
}

/* Initialize |grid| with alive cells with density p */
void init(cell_t *grid, float p)
{
    int i, j;
    for (i = TOP; i < BOTTOM + 1; i++) {
        for (j = LEFT; j < RIGHT + 1; j++) {
            grid[IDX(i,j)] = (rand() / (double) RAND_MAX < p);
        }
    }
}

/* Write |grid| to file |fname| in pbm (portable bitmap) format. */
void write_pbm(cell_t *grid, const char* fname)
{
    int i, j;
    FILE *f = fopen(fname, "w");
    if (!f) {
        printf("Cannot open %s for writing\n", fname);
        abort();
    }
    fprintf(f, "P1\n");
    fprintf(f, "# produced by game-of-life.c\n");
    fprintf(f, "%d %d\n", SIZE, SIZE);
    for (i = TOP; i < BOTTOM + 1; i++) {
        for (j = LEFT; j < RIGHT + 1; j++) {
            fprintf(f, "%d ", grid[IDX(i, j)]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

#define STRBUFSZ 128

int main(int argc, char *argv[])
{
    int s, nsteps;
    char fname[STRBUFSZ];
    const size_t GRID_SIZE = (SIZE + 2) * (SIZE + 2) * sizeof(cell_t);
    cell_t *cur = (cell_t *) malloc(GRID_SIZE);
    cell_t *next = (cell_t *) malloc(GRID_SIZE);

    srand(1234); /* init RNG */

    if (argc > 2) {
        printf("Usage: %s [nsteps]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 2) {
        nsteps = atoi(argv[1]);
    } else {
        nsteps = 256;
    }

    init(cur, 0.3);
    for (s = 0; s < nsteps; s++) {
        cell_t *tmp;
        snprintf(fname, STRBUFSZ, "gol%05d.pbm", s);
        write_pbm(cur, fname);
        copy_sides(cur);
        step(cur, next);
        /* swap |cur| and |next| */
        tmp = cur;
        cur = next;
        next = tmp;
    }
    return EXIT_SUCCESS;
}
