/****************************************************************************
 *
 * mpi-my-bcast.c - Broadcast using point-to-point communications
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

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void my_Bcast(int *v)
{
    int my_rank, comm_sz;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    if (my_rank > 0) {
        MPI_Recv(v,                /* buf          */
                 1,                /* count        */
                 MPI_INT,          /* datatype     */
                 (my_rank-1)/2,    /* source       */
                 0,                /* tag          */
                 MPI_COMM_WORLD,   /* comm         */
                 MPI_STATUS_IGNORE /* status       */
                 );
    }
    const int dest1 = (2 * my_rank + 1 < comm_sz ? 2 * my_rank + 1 : MPI_PROC_NULL);
    const int dest2 = (2 * my_rank + 2 < comm_sz ? 2 * my_rank + 2 : MPI_PROC_NULL);
    /* sending a message to MPI_PROC_NULL has no effect (see man page
       for MPI_Send) */
    MPI_Send(v, 1, MPI_INT, dest1, 0, MPI_COMM_WORLD);
    MPI_Send(v, 1, MPI_INT, dest2, 0, MPI_COMM_WORLD);
}

/**
 * Same as above, but using non-blocking send.
 */
void my_Ibcast(int *v)
{
    MPI_Request req[2];
    int my_rank, comm_sz;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    if (my_rank > 0) {
        MPI_Recv(v,                /* buf          */
                 1,                /* count        */
                 MPI_INT,          /* datatype     */
                 (my_rank-1)/2,    /* source       */
                 0,                /* tag          */
                 MPI_COMM_WORLD,   /* comm         */
                 MPI_STATUS_IGNORE /* status       */
                 );
    }
    const int dest1 = (2 * my_rank + 1 < comm_sz ? 2 * my_rank + 1 : MPI_PROC_NULL);
    const int dest2 = (2 * my_rank + 2 < comm_sz ? 2 * my_rank + 2 : MPI_PROC_NULL);
    /* sending a message to MPI_PROC_NULL has no effect */
    MPI_Isend(v, 1, MPI_INT, dest1, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Isend(v, 1, MPI_INT, dest2, 0, MPI_COMM_WORLD, &req[1]);
    /* Wait for all pending requests to complete */
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
}

int main(int argc, char *argv[])
{
    const int SENDVAL = 123; /* value to be broadcasted */
    int my_rank;
    int v;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* only process 0 sets `v` to the value to be broadcasted. */
    if (my_rank == 0) {
        v = SENDVAL;
    } else {
        v = -1;
    }

    printf("BEFORE: value of `v` at rank %d = %d\n", my_rank, v);
    my_Bcast(&v);

    if (v == SENDVAL) {
        printf("OK: ");
    } else {
        printf("ERROR: ");
    }
    printf("value of `v` at rank %d = %d\n", my_rank, v);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
