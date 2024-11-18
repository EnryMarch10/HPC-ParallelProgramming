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
double dot(const double *x, const double *y, int n)
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

    double *local_x, *local_y;

#ifdef USE_SCATTERV
    /* This block of code uses MPI_Scatterv() to distribute the input
       to the workers. To compile this version, use:

       mpicc -DUSE_SCATTERV -std=c99 -Wall -Wpedantic mpi-dot.c -o mpi.dot

    */

    if (my_rank == 0) {
        printf("Computing dot product using MPI_Scatterv()\n");
    }

    int *sendcounts = (int *) malloc(comm_sz * sizeof(*sendcounts));
    assert(sendcounts != NULL);
    int *displs = (int *) malloc(comm_sz * sizeof(*displs));
    assert(displs != NULL);
    for (i = 0; i < comm_sz; i++) {
        /* First, compute the starting and ending position of each
           block (iend is actually one element _past_ the ending
           position) */
        const int istart = n * i / comm_sz;
        const int iend = n * (i + 1) / comm_sz;
        const int blklen = iend - istart;
        sendcounts[i] = blklen;
        displs[i] = istart;
    }

    const int local_n = sendcounts[my_rank]; /* how many elements I must handle */

    /* All nodes (including the master) allocate the local vectors */
    local_x = (double *) malloc(local_n * sizeof(*local_x));
    assert(local_x != NULL);
    local_y = (double *) malloc(local_n * sizeof(*local_y));
    assert(local_y != NULL);

    /* Scatter vector x */
    MPI_Scatterv(x,             /* sendbuf             */
                 sendcounts,    /* sendcounts          */
                 displs,        /* displacements       */
                 MPI_DOUBLE,    /* sent datatype       */
                 local_x,       /* recvbuf             */
                 local_n,       /* recvcount           */
                 MPI_DOUBLE,    /* received datatype   */
                 0,             /* source              */
                 MPI_COMM_WORLD /* communicator        */
                 );

    /* Scatter vector y*/
    MPI_Scatterv(y,             /* sendbuf             */
                 sendcounts,    /* sendcounts          */
                 displs,        /* displacements       */
                 MPI_DOUBLE,    /* sent datatype       */
                 local_y,       /* recvbuf             */
                 local_n,       /* recvcount           */
                 MPI_DOUBLE,    /* received datatype   */
                 0,             /* source              */
                 MPI_COMM_WORLD /* communicator        */
                 );

    /* All nodes compute the local result */
    double local_result = dot(local_x, local_y, local_n);

    /* Reduce (sum) the local dot products */
    MPI_Reduce(&local_result,  /* send buffer          */
               &result,        /* receive buffer       */
               1,              /* count                */
               MPI_DOUBLE,     /* datatype             */
               MPI_SUM,        /* operation            */
               0,              /* destination          */
               MPI_COMM_WORLD  /* communicator         */
               );

    free(sendcounts);
    free(displs);
#else
    /* This block of code uses MPI_Scatter() to distribute the input
       to the workers; the master takes care of any leftover. To
       compile this version use:

       mpicc -std=c99 -Wall -Wpedantic mpi-dot.c -o mpi-dot

    */

    if (my_rank == 0) {
        printf("Computing dot product using MPI_Scatter()\n");
    }

    /* This version works for any value of n; the root takes care of
       the leftovers later on */
    const int local_n = n / comm_sz;

    /* All nodes (including the master) allocate the local vectors */
    local_x = (double *) malloc(local_n * sizeof(*local_x));
    assert(local_x != NULL);
    local_y = (double *) malloc(local_n * sizeof(*local_y));
    assert(local_y != NULL);

    /* Scatter vector x */
    MPI_Scatter(x,             /* sendbuf      */
                local_n,       /* count; how many elements to send to _each_ destination */
                MPI_DOUBLE,    /* sent datatype */
                local_x,       /* recvbuf      */
                local_n,       /* recvcount    */
                MPI_DOUBLE,    /* received datatype */
                0,             /* source       */
                MPI_COMM_WORLD /* communicator */
                );

    /* Scatter vector y*/
    MPI_Scatter(y,             /* sendbuf      */
                local_n,       /* count; how many elements to send to _each_ destination */
                MPI_DOUBLE,    /* sent datatype */
                local_y,       /* recvbuf      */
                local_n,       /* recvcount    */
                MPI_DOUBLE,    /* received datatype */
                0,             /* source       */
                MPI_COMM_WORLD /* communicator */
                );

    /* All nodes compute the local result */
    double local_result = dot(local_x, local_y, local_n);

    /* Reduce (sum) the local dot products */
    MPI_Reduce(&local_result,  /* send buffer          */
               &result,        /* receive buffer       */
               1,              /* count                */
               MPI_DOUBLE,     /* datatype             */
               MPI_SUM,        /* operation            */
               0,              /* destination          */
               MPI_COMM_WORLD  /* communicator         */
               );

    if (my_rank == 0) {
        /* the master handles the leftovers, if any */
        for (i = local_n * comm_sz; i < n; i++) {
            result += x[i] * y[i];
        }
    }
#endif
    free(local_x);
    free(local_y);

    if (my_rank == 0) {
        printf("Dot product: %f\n", result);
        if (fabs(result - n) < TOL) {
            printf("Check OK\n");
        } else {
            printf("Check failed: got %f, expected %f\n", result, (double)n);
        }
    }

    free(x); /* if x == NULL, does nothing */
    free(y);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
