/****************************************************************************
 *
 * mpi-ring.c - Ring communication with MPI
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
#include <mpi.h>

int main(int argc, char *argv[])
{
    int my_rank, comm_sz, K = 10;
    int val;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc > 1) {
        K = atoi(argv[1]);
    }
    const int prev = (my_rank - 1 + comm_sz) % comm_sz;
    const int next = (my_rank + 1) % comm_sz;

    if (my_rank == 0) {
        val = 1;
        MPI_Send(&val, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    for (int k = 0; k < K; k++) {
        MPI_Recv(&val,                 /* buf          */
                 1,                    /* count        */
                 MPI_INT,              /* datatype     */
                 prev,                 /* source       */
                 MPI_ANY_TAG,          /* tag          */
                 MPI_COMM_WORLD,       /* communicator */
                 MPI_STATUS_IGNORE     /* status       */
                 );
        if (my_rank != 0 || k < K - 1) {
            val++;
            MPI_Send(&val,             /* buf          */
                     1,                /* count        */
                     MPI_INT,          /* datatype     */
                     next,             /* dest         */
                     0,                /* tag          */
                     MPI_COMM_WORLD    /* communicator */
                     );
        }
    }

    if (my_rank == 0) {
        const int expected = comm_sz * K;
        printf("expected=%d, received=%d\n", expected, val);
        if (expected == val) {
            printf("Test OK\n");
        } else {
            printf("Test FAILED: expected value %d at rank 0, got %d\n", expected, val);
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
