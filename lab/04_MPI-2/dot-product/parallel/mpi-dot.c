#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs() */
#include <assert.h>
#include <mpi.h>

/*
 * Compute sum { x[i] * y[i] }, i=0, ... n-1
 */
double dot(const double* x, const double* y, int n)
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

    const int my_start = n * my_rank / comm_sz;
    const int my_end = n * (my_rank + 1) / comm_sz;
    const int my_n = my_end - my_start;

    int *sendcounts = NULL;
    int *displs = NULL;
    if (my_rank == 0) {
        sendcounts = (int *) malloc(comm_sz * sizeof(*sendcounts));
        assert(sendcounts != NULL);
        displs = (int *) malloc(comm_sz * sizeof(*displs));
        assert(displs != NULL);

        sendcounts[0] = my_n;
        displs[0] = my_start;
        for (i = 1; i < comm_sz; i++) {
            const int start = n * i / comm_sz;
            const int end = n * (i + 1) / comm_sz;
            sendcounts[i] = end - start;
            displs[i] = start;
        }
    }
    double *my_x = (double *) malloc(my_n * sizeof(*my_x));
    assert(my_x != NULL);

    /* Scatter x */
    MPI_Scatterv(x,             /* sendbuf    */
                 sendcounts,    /* sendcounts */
                 displs,        /* displs     */
                 MPI_DOUBLE,    /* sendtype   */
                 my_x,          /* recvbuf    */
                 my_n,          /* recvcount  */
                 MPI_DOUBLE,    /* recvtype   */
                 0,             /* root       */
                 MPI_COMM_WORLD /* comm       */
                 );

    double *my_y = (double *) malloc(my_n * sizeof(*my_y));
    assert(my_y != NULL);

    /* Scatter y */
    MPI_Scatterv(y,             /* sendbuf    */
                 sendcounts,    /* sendcounts */
                 displs,        /* displs     */
                 MPI_DOUBLE,    /* sendtype   */
                 my_y,          /* recvbuf    */
                 my_n,          /* recvcount  */
                 MPI_DOUBLE,    /* recvtype   */
                 0,             /* root       */
                 MPI_COMM_WORLD /* comm       */
                 );

    free(x);
    free(y);
    free(sendcounts);
    free(displs);

    const double local_result = dot(my_x, my_y, my_n);

    free(my_x);
    free(my_y);

    MPI_Reduce(&local_result, /* sendbuf  */
               &result,       /* recvbuf  */
               1,             /* count    */
               MPI_DOUBLE,    /* datatype */
               MPI_SUM,       /* op       */
               0,             /* root     */
               MPI_COMM_WORLD /* comm     */
               );

    if (my_rank == 0) {
        printf("Dot product: %f\n", result);
        if (fabs(result - n) < TOL) {
            printf("Check OK\n");
        } else {
            printf("Check failed: got %f, expected %f\n", result, (double) n);
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
