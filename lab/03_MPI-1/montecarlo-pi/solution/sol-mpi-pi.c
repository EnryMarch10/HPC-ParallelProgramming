/****************************************************************************
 *
 * mpi-pi.c - Monte Carlo approximatino of PI
 *
 * Copyright (C) 2017--2022, 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> /* for rand() */
#include <time.h>   /* for time() */
#include <math.h>   /* for fabs() */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Generate `n` random points within the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1 */
int generate_points(int n)
{
    int n_inside = 0;
    for (int i = 0; i < n; i++) {
        const double x = (rand() / (double) RAND_MAX * 2.0) - 1.0;
        const double y = (rand() / (double) RAND_MAX * 2.0) - 1.0;
        if (x * x + y * y < 1.0) {
            n_inside++;
        }
    }
    return n_inside;
}

int main(int argc, char *argv[])
{
    int my_rank, comm_sz;
    int inside = 0, n_points = 1000000;
    double pi_approx;
    int local_n, local_inside;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc > 1) {
        n_points = atoi(argv[1]);
    }

    /* Each process initializes the pseudo-random number generator; if
       we don't do this (or something similar), each process would
       produce the exact same sequence of pseudo-random numbers! */
    srand(my_rank * 11 + 7);

    local_n = n_points / comm_sz;

    /* The master handles the leftovers */
    if (my_rank == 0) {
        local_n += n_points % comm_sz;
    }

    /* All processes compute how many points are inside the circle */
    printf("Proc %d generates %d points...\n", my_rank, local_n);
    local_inside = generate_points(local_n);

    /* The solution below is NOT efficient since it relies on
       send/receive operations to accumulate the values at the
       master. The correct solution is to use MPI_Reduce() */
    if (my_rank > 0) {
        /* All processes, except the master, send the local count to
           proc 0 */
        MPI_Send(&local_inside,        /* buf          */
                 1,                    /* count        */
                 MPI_INT,              /* datatype     */
                 0,                    /* dest         */
                 0,                    /* tag          */
                 MPI_COMM_WORLD        /* communicator */
                 );
    } else {
        /* The master performs the reduction */
        inside = local_inside;
        int tmp;
        for (int i = 1; i < comm_sz; i++) {
            MPI_Recv(&tmp,             /* buf          */
                     1,                /* count        */
                     MPI_INT,          /* datatype     */
                     MPI_ANY_SOURCE,   /* source       */
                     MPI_ANY_TAG,      /* tag          */
                     MPI_COMM_WORLD,   /* communicator */
                     MPI_STATUS_IGNORE /* status       */
                     );
            inside += tmp;
        }
    }
    if (my_rank == 0) {
        pi_approx = 4.0 * inside / (double) n_points;
        printf("PI approximation is %f (true value=%f, rel error=%.3f%%)\n", pi_approx, M_PI, 100.0*fabs(pi_approx - M_PI) / M_PI);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
