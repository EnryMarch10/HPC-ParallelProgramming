/****************************************************************************
 *
 * mpi-type-struct.c - Simple demo of the MPI_Type_create_struct function
 *
 * Based on https://computing.llnl.gov/tutorials/mpi/#Derived_Data_Types
 *
 * Copyright (C) 2017, 2021 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 *      mpicc -std=c99 -Wall -Wpedantic mpi-type-struct.c -o mpi-type-struct
 *
 * Run with:
 *
 *      mpirun -n 4 ./mpi-type-struct
 *
 ****************************************************************************/

#include <stdio.h>
#include <mpi.h>

#define NELEM 25

int main(int argc, char *argv[])
{
    int my_rank, comm_sz, i;

    typedef struct {
        float x, y, z;
        float velocity;
        int  n, type;
    } particle_t;

    particle_t   particles[NELEM];
    MPI_Datatype particletype, oldtypes[2];
    int          blklens[2];
    MPI_Aint     displs[2], lb, extent;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    oldtypes[0] = MPI_FLOAT;
    blklens[0]= 4;
    displs[0] = 0;

    MPI_Type_get_extent(MPI_FLOAT, &lb, &extent);
    oldtypes[1] = MPI_INT;
    blklens[1] = 2;
    displs[1] = 4 * extent;

    /* define structured type and commit it */
    MPI_Type_create_struct(2,                  /* count                     */
                           blklens,            /* array of blocklen         */
                           displs,             /* array of displacements    */
                           oldtypes,           /* array of types            */
                           &particletype);

    MPI_Type_commit(&particletype);

    /* task 0 initializes the particle array and then sends it to each task */
    if (0 == my_rank) {
        for (i=0; i<NELEM; i++) {
            particles[i].x = i * 1.0;
            particles[i].y = i * -1.0;
            particles[i].z = i * 1.0;
            particles[i].velocity = 0.25;
            particles[i].n = i;
            particles[i].type = i % 2;
        }
    }

    MPI_Bcast(particles,        /* buffer       */
              NELEM,            /* count        */
              particletype,     /* datatype     */
              0,                /* source       */
              MPI_COMM_WORLD    /* comm         */
              );

    printf("rank= %d   %3.2f %3.2f %3.2f %3.2f %d %d\n", my_rank,
           particles[3].x, particles[3].y, particles[3].z,
           particles[3].velocity,
           particles[3].n, particles[3].type);

    /* free datatype when done using it */
    MPI_Type_free(&particletype);
    MPI_Finalize();
    return 0;
}
