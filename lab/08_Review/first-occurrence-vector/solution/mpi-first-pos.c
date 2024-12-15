/****************************************************************************
 *
 * mpi-first-pos.c - First occurrence of a value in a vector
 *
 * Copyright (C) 2022--2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
#include <mpi.h>

int main(int argc, char *argv[])
{
    int my_rank, comm_sz, N, k, minpos;
    int *v = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc > 1) {
        N = atoi(argv[1]);
    } else {
        N = comm_sz * 10;
    }

    if (argc > 2) {
        k = atoi(argv[2]);
    } else {
        k = N/8;
    }

    if (N % comm_sz != 0 && my_rank == 0) {
        fprintf(stderr, "FATAL: array length (%d) must be a multiple of comm_sz (%d)\n", N, comm_sz);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* The array is initialized by process 0 */
    if (my_rank == 0) {
        printf("N=%d, k=%d\n", N, k);
        v = (int *) malloc(N * sizeof(*v));
        assert(v != NULL);
        for (int i = 0; i < N; i++) {
            v[i] = i % (N / 4);
        }
    }

    /* All processes initialize the local buffers */
    const int local_N = N / comm_sz;
    int *local_v = (int *) malloc(local_N * sizeof(*local_v));
    assert(local_v != NULL);

    /* The master distributes `v[]` to the other processes */
    MPI_Scatter(v,             /* senfbuf      */
                local_N,       /* sendcount    */
                MPI_INT,       /* sendtype     */
                local_v,       /* recvbuf      */
                local_N,       /* recvcount    */
                MPI_INT,       /* recvtype     */
                0,             /* root         */
                MPI_COMM_WORLD /* comm         */
                );

    /* Every process performs a local sequential search on the local
       portion of `v[]`. There are two problems: (i) all local indices
       must be mapped to the corresponding global indices; (ii) since
       the result is computed as the min-reduction of the partial
       results, if a process does not find the key on the local array,
       it must send `N` to process 0. */
    int i = 0, local_minpos;
    while (i < local_N && local_v[i] != k) {
        i++;
    }
    if (i < local_N) {
        local_minpos = my_rank * local_N + i; /* map local indices to global indices */
    } else {
        local_minpos = N;
    }

    /* Performs a min-reduction of the local results */
    MPI_Reduce(&local_minpos,   /* sendbuf      */
               &minpos,         /* recvbuf      */
               1,               /* count        */
               MPI_INT,         /* datatype     */
               MPI_MIN,         /* op           */
               0,               /* root         */
               MPI_COMM_WORLD );

    free(local_v);

    if (my_rank == 0) {
        const int expected = k >= 0 && k < (N / 4) ? k : N;
        printf("Result: %d ", minpos);
        if (minpos == expected) {
            printf("OK\n");
        } else {
            printf("FAILED (expected %d)\n", expected);
        }
    }

    free(v);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
