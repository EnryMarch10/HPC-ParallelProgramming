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
void step(cell_t *cur, cell_t *next, int ext_n)
{
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    for (int i = LEFT; i <= RIGHT; i++) {
        const cell_t left   = cur[i - 1];
        const cell_t center = cur[i];
        const cell_t right  = cur[i + 1];
        next[i] =
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
    cell_t *cur, *next;

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
    const int LEFT_GHOST = 0;
    const int LEFT = 1;
    const int RIGHT_GHOST = ext_width - 1;
    const int RIGHT = RIGHT_GHOST - 1;
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
    next = (cell_t *) malloc(ext_size);
    assert(next != NULL);

    /* Initialize the domain */
    init_domain(cur, ext_width);

    /* Evolve the CA */
    for (int s = 0; s < steps; s++) {
        /* Dump the current state */
        dump_state(out, cur, ext_width);

        /* Fill ghost cells */
        cur[RIGHT_GHOST] = cur[LEFT];
        cur[LEFT_GHOST] = cur[RIGHT];

        /* Compute next state */
        step(cur, next, ext_width);

        /* swap cur and next */
        cell_t *tmp = cur;
        cur = next;
        next = tmp;
    }

    free(cur);
    free(next);

    fclose(out);

    return EXIT_SUCCESS;
}
