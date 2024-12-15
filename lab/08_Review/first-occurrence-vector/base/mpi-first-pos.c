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
        k = N / 8;
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

    /* [TODO] replace this block with a true parallel version */
    if (my_rank == 0) {
        minpos = 0;
        while (minpos < N && v[minpos] != k) {
            minpos++;
        }
        /* Invariant: (minpos == N) or (minpos is the first occurrence of k in v[]) */
    }

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
