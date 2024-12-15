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

    const int local_N = N / comm_sz;
    int *local_v = (int *) malloc(local_N * sizeof(*local_v));
    assert(local_v != NULL);

    MPI_Scatter(v,             /* const void *sendbuf */
                local_N,       /* int sendcount */
                MPI_INT,       /* MPI_Datatype sendtype */
                local_v,       /* void *recvbuf */
                local_N,       /* int recvcount */
                MPI_INT,       /* MPI_Datatype recvtype */
                0,             /* int root */
                MPI_COMM_WORLD /* MPI_Comm comm */
                );

    int local_minpos = N;
    for (int i = 0; i < local_N; i++) {
        if (local_v[i] == k) {
            local_minpos = i + local_N * my_rank;
            break;
        }
    }

    MPI_Reduce(&local_minpos, /* const void *sendbuf */
               &minpos,       /* void *recvbuf */
               1,             /* int count */
               MPI_INT,       /* MPI_Datatype datatype */
               MPI_MIN,       /* MPI_Op op */
               0,             /* int root */
               MPI_COMM_WORLD /* MPI_Comm comm */
               );

    if (my_rank == 0) {
        const int expected = k >= 0 && k < (N / 4) ? k : N;
        printf("Result: %d ", minpos);
        if (minpos == expected) {
            printf("OK\n");
        } else {
            printf("FAILED (expected %d)\n", expected);
        }
    }

    free(local_v);
    free(v);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
