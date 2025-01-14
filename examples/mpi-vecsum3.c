/****************************************************************************
 *
 * mpi-vecsum.c - Parallel vector sum using MPI.
 *
 * Copyright (C) 2018, 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * Parallel vector sum using MPI. This version shows how MPI_Scatterv
 * and MPI_Gatherv can be used to scatter/gather irregular blocks.
 *
 * Compile with:
 *
 *      mpicc -std=c99 -Wall -Wpedantic mpi-vecsum3.c -o mpi-vecsum3
 *
 * Run with:
 *
 *      mpirun -n 4 ./mpi-vecsum3 [n]
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs() */
#include <mpi.h>

/*
 * Compute z[i] = x[i] + y[i], i=0, ... n-1
 */
void sum(double* x, double* y, double* z, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}

int main(int argc, char* argv[])
{
    double *x, *local_x, *y, *local_y, *z, *local_z;
    int *counts, *displs;
    int n = 1000, local_n, i;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc == 2) {
        n = atoi(argv[1]);
    }

    /* Every process computes `sendcounts` and `displs`; actually,
       only the master needs this information so we are wasting memory
       on the other processes, but this makes the program simpler. */
    counts = (int *) malloc(comm_sz * sizeof(*counts));
    displs = (int *) malloc(comm_sz * sizeof(*displs));
    for (i = 0; i < comm_sz; i++) {
        const int start = n * i / comm_sz;
        const int end = n * (i + 1) / comm_sz;
        counts[i] = end - start;
        displs[i] = start;
    }

    local_n = counts[my_rank];

    x = y = z = NULL;

    if (my_rank == 0) {
        /* Only the master allocates the vectors */
        x = (double *) malloc(n * sizeof(*x));
        y = (double *) malloc(n * sizeof(*y));
        z = (double *) malloc(n * sizeof(*z));
        for (i = 0; i < n; i++) {
            x[i] = i;
            y[i] = n - 1 - i;
        }
    }

    /* All nodes (including the master) allocate the local vectors.
       Note that `local_n` may be different from each node, if `n` is
       not a multiple of `comm_sz`. */
    local_x = (double *) malloc(local_n * sizeof(*local_x));
    local_y = (double *) malloc(local_n * sizeof(*local_y));
    local_z = (double *) malloc(local_n * sizeof(*local_z));

    /* Scatter `x[]` */
    MPI_Scatterv(x,             /* sendbuf */
                 counts,        /* sendcounts */
                 displs,        /* displacements */
                 MPI_DOUBLE,    /* sent MPI_Datatype */
                 local_x,       /* recvbuf */
                 local_n,       /* recvcount */
                 MPI_DOUBLE,    /* received MPI_Datatype */
                 0,             /* root */
                 MPI_COMM_WORLD /* communicator */
                 );

    /* Scatter `y[]` */
    MPI_Scatterv(y,             /* sendbuf */
                 counts,        /* sendcounts */
                 displs,        /* displacements */
                 MPI_DOUBLE,    /* sent MPI_Datatype */
                 local_y,       /* recvbuf */
                 local_n,       /* recvcount */
                 MPI_DOUBLE,    /* received MPI_Datatype */
                 0,             /* root */
                 MPI_COMM_WORLD /* communicator */
                 );

    /* All nodes compute the local result */
    sum(local_x, local_y, local_z, local_n);

    /* Gather results from all nodes */
    MPI_Gatherv(local_z,       /* sendbuf */
                local_n,       /* sendcount */
                MPI_DOUBLE,    /* sendtype */
                z,             /* recvbuf */
                counts,        /* receive counts */
                displs,        /* displacements */
                MPI_DOUBLE,    /* recvtype */
                0,             /* root (where to send) */
                MPI_COMM_WORLD /* communicator */
                );

    /* Enable to print the result */
#if 0
    if (my_rank == 0) {
        for (i = 0; i < n; i++) {
            printf("z[%d] = %f\n", i, z[i]);
        }
    }
#endif

    /* The master checks the result */
    if (my_rank == 0) {
        for (i = 0; i < n; i++) {
            if (fabs(z[i] - (n - 1)) > 1e-6) {
                fprintf(stderr, "Test FAILED: z[%d]=%f, expected %f\n", i, z[i], (double) (n - 1));
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        printf("Test OK\n");
    }

    free(x); /* If x == NULL, no operation is performed */
    free(y);
    free(z);

    free(local_x);
    free(local_y);
    free(local_z);

    free(counts);
    free(displs);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
