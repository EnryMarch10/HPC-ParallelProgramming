/****************************************************************************
 *
 * mpi-mandelbrot.c - Mandelbrot set
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
#include <stdint.h>
#include <assert.h>
#include <mpi.h>

const int MAX_IT = 100;

/* The __attribute__(( ... )) definition is gcc-specific, and tells
   the compiler that the fields of this structure should not be padded
   or aligned in any way. */
typedef struct __attribute__((__packed__)) {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

/* color gradient from https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia */
const pixel_t colors[] = {
    { 66,  30,  15}, /* r, g, b */
    { 25,   7,  26},
    {  9,   1,  47},
    {  4,   4,  73},
    {  0,   7, 100},
    { 12,  44, 138},
    { 24,  82, 177},
    { 57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201,  95},
    {255, 170,   0},
    {204, 128,   0},
    {153,  87,   0},
    {106,  52,   3} };
const int NCOLORS = sizeof(colors)/sizeof(colors[0]);

/*
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + cx + i*cy;
 *
 * Returns the first `n` such that `z_n > bound`, or `MAX_IT` if `z_n` is below
 * `bound` after `MAX_IT` iterations.
 */
int iterate(float cx, float cy)
{
    float x = 0.0f, y = 0.0f, x_new, y_new;
    int it;
    for (it = 0; it < MAX_IT && x * x + y * y <= 2.0 * 2.0; it++) {
        x_new = x * x - y * y + cx;
        y_new = 2.0 * x * y + cy;
        x = x_new;
        y = y_new;
    }
    return it;
}

/* Draw the rows of the Mandelbrot set from `y_start` (inclusive) to
   `y_end` (excluded) to the bitmap pointed to by `p`. Note that `p`
   must point to the beginning of the bitmap where the portion of
   image will be stored; in other words, this function writes to
   pixels p[0], p[1], ... `xsize` and `y_size` MUST be the sizes
   of the WHOLE image. */
void draw_lines(int y_start, int y_end, pixel_t* p, int xsize, int y_size)
{
    int x, y;
    for (y = y_start; y < y_end; y++) {
        for (x = 0; x < xsize; x++) {
            const float cx = -2.5 + 3.5 * (float) x / (xsize - 1);
            const float cy = 1 - 2.0 * (float) y / (y_size - 1);
            const int v = iterate(cx, cy);
            if (v < MAX_IT) {
                p->r = colors[v % NCOLORS].r;
                p->g = colors[v % NCOLORS].g;
                p->b = colors[v % NCOLORS].b;
            } else {
                p->r = p->g = p->b = 0;
            }
            p++;
        }
    }
}

int main(int argc, char *argv[])
{
    int my_rank, comm_sz;
    FILE *out = NULL;
    const char* fname = "mandelbrot.ppm";
    pixel_t *bitmap = NULL;
    int xsize, y_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc > 1) {
        y_size = atoi(argv[1]);
    } else {
        y_size = 1024;
    }

    xsize = y_size * 1.4;

    /* xsize and y_size are known to all processes */
    if (my_rank == 0) {
        out = fopen(fname, "w");
        if (!out) {
            fprintf(stderr, "Error: cannot create %s\n", fname);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Write the header of the output file */
        fprintf(out, "P6\n");
        fprintf(out, "%d %d\n", xsize, y_size);
        fprintf(out, "255\n");

        /* Allocate the complete bitmap */
        bitmap = (pixel_t *) malloc(xsize * y_size * sizeof(*bitmap));
        assert(bitmap != NULL);
    }

#ifdef USE_GATHERV
    /* This version makes use of MPI_Gatherv to collect portions of
       different sizes. To compile this version, use:

       mpicc -std=c99 -Wall -Wpedantic -DUSE_GATHERV mpi-mandelbrot.c -o mpi-mandelbrot

    */
    int y_start[comm_sz], y_end[comm_sz], counts[comm_sz], displs[comm_sz];
    for (int i = 0; i < comm_sz; i++) {
        y_start[i] = y_size * i / comm_sz;
        y_end[i] = y_size * (i + 1) / comm_sz;
        /* counts[] and displs[] must be measured as the number of
           "array elements", NOT bytes; however, in this case the type
           of array elements that are gathered together is MPI_BYTE
           (see MPI_Gatherv below), so we need to multiply by
           sizeof(pixel_t) */
        counts[i] = (y_end[i] - y_start[i]) * xsize * sizeof(pixel_t);
        displs[i] = y_start[i] * xsize * sizeof(pixel_t);
    }

    pixel_t *local_bitmap = (pixel_t *) malloc(counts[my_rank]);
    assert(local_bitmap != NULL);

    const double tstart = MPI_Wtime();

    draw_lines(y_start[my_rank], y_end[my_rank], local_bitmap, xsize, y_size);

    MPI_Gatherv(local_bitmap,    /* sendbuf      */
                counts[my_rank], /* sendcount    */
                MPI_BYTE,        /* datatype     */
                bitmap,          /* recvbuf      */
                counts,          /* recvcounts[] */
                displs,          /* displacements[] */
                MPI_BYTE,        /* datatype     */
                0,               /* root         */
                MPI_COMM_WORLD
                );

    const double elapsed = MPI_Wtime() - tstart;

    if (my_rank == 0) {
        fwrite(bitmap, sizeof(*bitmap), xsize * y_size, out);
        fclose(out);

        printf("Elapsed time (s): %f\n", elapsed);
    }
    free(bitmap);
    free(local_bitmap);
#else
    const int local_y_size = y_size / comm_sz;
    const int y_start = local_y_size * my_rank;
    const int y_end = local_y_size * (my_rank + 1);
    pixel_t *local_bitmap = (pixel_t *) malloc(xsize * local_y_size * sizeof(*local_bitmap));
    assert(local_bitmap != NULL);

    const double tstart = MPI_Wtime();

    draw_lines(y_start, y_end, local_bitmap, xsize, y_size);

    MPI_Gather(local_bitmap,             /* sendbuf      */
               xsize * local_y_size * 3, /* sendcount    */
               MPI_BYTE,                 /* datatype     */
               bitmap,                   /* recvbuf      */
               xsize * local_y_size * 3, /* recvcount    */
               MPI_BYTE,                 /* datatype     */
               0,                        /* root         */
               MPI_COMM_WORLD
               );

    if (my_rank == 0) {
        /* the master computes the last (y_size % comm_sz) lines of the image */
        if (y_size % comm_sz) {
            const int skip = local_y_size * comm_sz; /* how many rows to skip */
            draw_lines(skip, y_size, &bitmap[skip * xsize], xsize, y_size);
        }
        const double elapsed = MPI_Wtime() - tstart;

        fwrite(bitmap, sizeof(*bitmap), xsize * y_size, out);
        fclose(out);

        printf("Elapsed time (s): %f\n", elapsed);
    }
    free(bitmap);
    free(local_bitmap);
#endif

    MPI_Finalize();
    return EXIT_SUCCESS;
}
