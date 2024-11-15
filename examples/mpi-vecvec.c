/****************************************************************************
 *
 * mpi-vecvec.c - Test MPI derived datatype
 *
 * Copyright 2021 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * Compile with:
 *
 *      mpicc -std=c99 -Wall -Wpedantic mpi-vecvec.c -o mpi-vecvec
 *
 * Run with:
 *
 *      mpirun -n 2 ./mpi-vecvec
 *
 ****************************************************************************/

#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    int my_rank, comm_sz;
    int count, blocklen, stride;
    MPI_Datatype vec, vecvec;
    int v[36];
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    count = 2; blocklen = 2; stride = 3;
    MPI_Type_vector(count, blocklen, stride, MPI_FLOAT, &vec);
    MPI_Type_commit(&vec);
    count = 2; blocklen = 1; stride = 3;
    MPI_Type_vector(count, blocklen, stride, vec, &vecvec);
    MPI_Type_commit(&vecvec);
    MPI_Aint lb, extent;
    MPI_Type_get_extent(vecvec, &lb, &extent);
    printf("lb=%d extent=%d\n", (int) lb, (int) extent);
    for (i = 0; i < 36; i++) {
        if (my_rank == 0) {
            v[i] = i;
        } else {
            v[i] = -1;
        }
    }

    /* v[] nel processo 0: [0, 1, 2, ... 35];

       v[] negli altri processi: [-1, -1, ... -1]; */

    if (my_rank == 0) {
        MPI_Send(v,             /* sendbuf      */
                 1,             /* count        */
                 vecvec,        /* datatype     */
                 1,             /* dest         */
                 0,             /* tag          */
                 MPI_COMM_WORLD);
    } else {
        MPI_Recv(v,             /* recvbuf      */
                 1,             /* recvcount    */
                 vecvec,        /* recvtype     */
                 0,             /* source       */
                 MPI_ANY_TAG,   /* tag          */
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        for (i = 0; i < 36; i++) {
            printf("[%2d]=%2d ", i, v[i]);
            if (i % 6 == 5) {
                printf("\n");
            }
        }
        printf("\n");
    }
    MPI_Type_free(&vecvec);
    MPI_Type_free(&vec);
    MPI_Finalize();
    return 0;
}
