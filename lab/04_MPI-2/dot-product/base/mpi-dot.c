/****************************************************************************
 *
 * mpi-dot.c - Dot product
 *
 * Copyright (C) 2016--2023 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
#include <math.h> /* for fabs() */
#include <assert.h>
#include <mpi.h>

/*
 * Compute sum { x[i] * y[i] }, i=0, ... n-1
 */
double dot(const double* x, const double* y, int n)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        s += x[i] * y[i];
    }
    return s;
}

int main(int argc, char *argv[])
{
    const double TOL = 1e-5;
    double *x = NULL, *y = NULL, result = 0.0;
    int i, n = 1000;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (my_rank == 0) {
        /* The master allocates the vectors */
        x = (double *) malloc(n * sizeof(*x));
        assert(x != NULL);
        y = (double *) malloc(n * sizeof(*y));
        assert(y != NULL);
        for (i = 0; i < n; i++) {
            x[i] = i + 1.0;
            y[i] = 1.0 / x[i];
        }
    }
    /* [TODO] This is not a true parallel version, since the master
       does everything */
    if (my_rank == 0) {
        result = dot(x, y, n);
    }

    if (my_rank == 0) {
        printf("Dot product: %f\n", result);
        if (fabs(result - n) < TOL) {
            printf("Check OK\n");
        } else {
            printf("Check failed: got %f, expected %f\n", result, (double) n);
        }
    }

    free(x); /* if x == NULL, does nothing */
    free(y);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
