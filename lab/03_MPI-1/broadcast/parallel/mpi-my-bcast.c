#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void my_Bcast(int *v)
{
    int my_rank, comm_sz;
    int dest1, dest2;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank > 0) {
        const int parent = (my_rank - 1) / 2;
        MPI_Recv(v,                /* buf      */
                 1,                /* count    */
                 MPI_INT,          /* datatype */
                 parent,           /* source   */
                 0,                /* tag      */
                 MPI_COMM_WORLD,   /* comm     */
                 MPI_STATUS_IGNORE /* status   */
                 );
    }
    dest1 = 2 * my_rank + 1 < comm_sz ? 2 * my_rank + 1 : MPI_PROC_NULL;
    dest2 = 2 * my_rank + 2 < comm_sz ? 2 * my_rank + 2 : MPI_PROC_NULL;
    MPI_Send(v,             /* buf      */
             1,             /* count    */
             MPI_INT,       /* datatype */
             dest1,         /* dest     */
             0,             /* tag      */
             MPI_COMM_WORLD /* comm     */
             );
    MPI_Send(v,             /* buf      */
             1,             /* count    */
             MPI_INT,       /* datatype */
             dest2,         /* dest     */
             0,             /* tag      */
             MPI_COMM_WORLD /* comm     */
             );
}

void my_Ibcast(int *v)
{
    int my_rank, comm_sz;
    int dest1, dest2;
    MPI_Request req[2];

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank > 0) {
        const int parent = (my_rank - 1) / 2;
        MPI_Recv(v,                /* buf      */
                 1,                /* count    */
                 MPI_INT,          /* datatype */
                 parent,           /* source   */
                 0,                /* tag      */
                 MPI_COMM_WORLD,   /* comm     */
                 MPI_STATUS_IGNORE /* status   */
                 );
    }
    dest1 = 2 * my_rank + 1 < comm_sz ? 2 * my_rank + 1 : MPI_PROC_NULL;
    dest2 = 2 * my_rank + 2 < comm_sz ? 2 * my_rank + 2 : MPI_PROC_NULL;
    MPI_Isend(v,              /* buf      */
              1,              /* count    */
              MPI_INT,        /* datatype */
              dest1,          /* dest     */
              0,              /* tag      */
              MPI_COMM_WORLD, /* comm     */
              &req[0]         /* request  */
              );
    MPI_Isend(v,              /* buf      */
              1,              /* count    */
              MPI_INT,        /* datatype */
              dest2,          /* dest     */
              0,              /* tag      */
              MPI_COMM_WORLD, /* comm     */
              &req[1]         /* request  */
              );
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE); /* Serve per permettere al chiamante la sovrascrittura del buffer v */
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

    my_Ibcast(&v);

    if (v == SENDVAL) {
        printf("OK: ");
    } else {
        printf("ERROR: ");
    }
    printf("value of `v` at rank %d = %d\n", my_rank, v);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
