/****************************************************************************
 *
 * mandelbrot.c - displays the Mandelbrot set using ASCII characters
 *
 * Copyright (C) 2017, 2020, 2021 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * This program computes and displays the Mandelbrot set using ASCII characters.
 * Since characters are displayed one at a time,
 *
 * Compile with
 *
 *      gcc -std=c99 -Wall -Wpedantic -fopenmp omp-mandelbrot-ordered.c -o omp-mandelbrot-ordered
 *
 * and run with:
 *
 *      OMP_NUM_THREADS=4 ./omp-mandelbrot-ordered
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>

const int MAXIT = 500000; /* A very large value to make the program slower */

/* Picture window size, in characters */
const int XSIZE = 78, YSIZE = 62;

/* Coordinates of the bounding box of the Mandelbrot set */
const double XMIN = -2.5, XMAX = 1.0;
const double YMIN = -1.0, YMAX = 1.0;

/*
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + cx + i*cy;
 *
 * Returns the first n such that z_n > |bound|, or |MAXIT| if z_n is below
 * |bound| after |MAXIT| iterations.
 */
int iterate(float cx, float cy)
{
    int it;
    float x = 0.0, y = 0.0, xnew, ynew;
    for ( it = 0; (it < MAXIT) && (x*x + y*y < 2*2); it++ ) {
        xnew = x * x - y * y + cx;
        ynew = 2.0f * x * y + cy;
        x = xnew;
        y = ynew;
    }
    return it;
}

int main(int argc, char *argv[])
{
    float tstart, elapsed;
    const char charset[] = ".,c8M@jawrpogOQEPGJ";

    tstart = hpc_gettime();
    /* x, y are private by default, being used in "for" statements
       that are both parallelized by the collapse(2) directive */
#pragma omp parallel for default(none) collapse(2) shared(XSIZE,YSIZE,XMIN,XMAX,YMIN,YMAX,charset,MAXIT) ordered
    for (int y = 0; y < YSIZE; y++) {
        for (int x = 0; x < XSIZE; x++) {
            const double cx = XMIN + (XMAX - XMIN) * (float)(x) / (XSIZE - 1);
            const double cy = YMAX - (YMAX - YMIN) * (float)(y) / (YSIZE - 1);
            const int v = iterate(cx, cy);
#pragma omp ordered
            {
                char c = ' ';
                if (v < MAXIT) {
                    c = charset[v % (sizeof(charset) - 1)];
                }
                putchar(c);
                if (x + 1 == XSIZE) {
                    puts("|");
                }
            }
        }
    }
    elapsed = hpc_gettime() - tstart;
    printf("Elapsed time %.2f\n", elapsed);
    return EXIT_SUCCESS;
}
