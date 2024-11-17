#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> /* for rand() */
#include <time.h>   /* for time() */
#include <math.h>   /* for fabs() */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEBUG

/* Generate `n` random points within the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1 */
long generate_points(long n)
{
    long n_inside = 0;
    for (long i = 0; i < n; i++) {
        const double x = (rand() / (double) RAND_MAX * 2.0) - 1.0;
        const double y = (rand() / (double) RAND_MAX * 2.0) - 1.0;
        if (x * x + y * y < 1.0) {
            n_inside++;
        }
    }
    return n_inside;
}

int main(int argc, char *argv[])
{
    int my_rank, comm_sz;
    long inside = 0, n_points = 1000000;
    double pi_approx;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc > 1) {
        n_points = atol(argv[1]);
    }

    if (n_points < 1) {
        fprintf(stderr, "FATAL: N points must be 1 or greater\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* --- BEFORE ---
     * This solution only works if n_points is a multiple of the
     * number of processes.
     */
    // if (n_points % comm_sz != 0) {
    //     fprintf(stderr, "FATAL: N points must be a multiple of the number of processes %d\n", comm_sz);
    //     MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    // }
    // const long my_n_points = n_points / comm_sz;

    /* --- AFTER ---
     * I chose to use the following formulas:
     * 1 - my_start = (n * my_id) / P;
     * 2 - my_end = (n * (my_id + 1)) / P;
     * where
     * - my_* is the index of the interval
     * - n is the number of elements
     * - my_id is the process id (from 0 to P - 1)
     * - P is the number of processes
     */
    const long my_n_points = n_points * (my_rank + 1) / comm_sz - n_points * my_rank / comm_sz;

    /* Each process initializes the pseudo-random number generator; if
       we don't do this (or something similar), each process would
       produce the exact same sequence of pseudo-random numbers! */
    srand(my_rank * 11 + 7);

#ifdef DEBUG
    printf("[%2d] My points to generate inside the square are = %ld\n", my_rank, my_n_points);
#endif
    inside = generate_points(my_n_points);
#ifdef DEBUG
    printf("[%2d] My generated points inside the circle are = %ld\n", my_rank, inside);
#endif

    long all_inside;
    MPI_Reduce(&inside,       /* sendbuf  */
               &all_inside,   /* recvbuf  */
               1,             /* count    */
               MPI_INT,       /* datatype */
               MPI_SUM,       /* op       */
               0,             /* root     */
               MPI_COMM_WORLD /* comm     */
    );

    if (my_rank == 0) {
#ifdef DEBUG
        printf("[ 0] Total generated points inside the circle are = %ld\n", all_inside);
#endif
        pi_approx = 4.0 * all_inside / (double) n_points;
        printf("PI approximation is %f (true value=%f, rel error=%.3f%%)\n", pi_approx, M_PI, 100.0 * fabs(pi_approx - M_PI) / M_PI);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
