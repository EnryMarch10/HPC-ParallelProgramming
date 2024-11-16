#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int my_rank, comm_sz, K = 10;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc > 1) {
        K = atoi(argv[1]);
    }

    if (K < 1) {
        fprintf(stderr, "FATAL: K must be 1 or greater\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int v;
    if (my_rank == 0) {
        v = 0;
        for (; K > 0; K--) {
            v++;
            MPI_Send(&v,                      /* buf      */
                     1,                       /* count    */
                     MPI_INT,                 /* datatype */
                     (my_rank + 1) % comm_sz, /* dest     */
                     0,                       /* tag      */
                     MPI_COMM_WORLD           /* comm     */
                     );
            MPI_Recv(&v,               /* buf      */
                     1,                /* count    */
                     MPI_INT,          /* datatype */
                     comm_sz - 1,      /* source   */
                     MPI_ANY_TAG,      /* tag      */
                     MPI_COMM_WORLD,   /* comm     */
                     MPI_STATUS_IGNORE /* status   */
                     );
        }
        printf("My rank is 0, my final v = %d\n", v);
    } else {
        for (; K > 0; K--) {
            MPI_Recv(&v,               /* buf      */
                     1,                /* count    */
                     MPI_INT,          /* datatype */
                     my_rank - 1,      /* source   */
                     MPI_ANY_TAG,      /* tag      */
                     MPI_COMM_WORLD,   /* comm     */
                     MPI_STATUS_IGNORE /* status   */
                     );
            v++;
            MPI_Send(&v,                      /* buf      */
                     1,                       /* count    */
                     MPI_INT,                 /* datatype */
                     (my_rank + 1) % comm_sz, /* dest     */
                     0,                       /* tag      */
                     MPI_COMM_WORLD           /* comm     */
                     );
        }
        printf("My rank is %d, my final v = %d\n", my_rank, v - 1);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
