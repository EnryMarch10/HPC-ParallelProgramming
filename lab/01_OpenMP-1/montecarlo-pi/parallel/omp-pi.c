/* The rand_r() function is available only if _XOPEN_SOURCE=600 */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs */
#include <omp.h>

/* Generate `n` random points within the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1. */
long unsigned int generate_points(long unsigned int n)
{
    /* [TODO] parallelize the body of this function */
    long unsigned int n_inside = 0;
    /* The C function `rand()` is not thread-safe, since it modifies a
       global seed; therefore, it can not be used inside a parallel
       region. We use `rand_r()` with an explicit per-thread
       seed. However, this means that in general the result computed
       by this program depends on the number of threads. */
#pragma omp parallel default(none) shared(n, n_inside)
    {
        unsigned int my_seed = 17 + 19 * omp_get_thread_num();
#pragma omp for reduction(+:n_inside)
        for (long unsigned int i = 0; i < n; i++) {
            /* Generate two random values in the range [-1, 1] */
            const double x = (2.0 * rand_r(&my_seed) / (double) RAND_MAX) - 1.0;
            const double y = (2.0 * rand_r(&my_seed) / (double) RAND_MAX) - 1.0;
            if (x * x + y * y <= 1.0) {
                n_inside++;
            }
        }
    }
    return n_inside;
}

int main(int argc, char *argv[])
{
    long unsigned int n_points = 10000;
    long unsigned int n_inside;
    const double PI_EXACT = 3.14159265358979323846;

    /* On my architecture 8 bytes */
    /* printf("Size: %lu bytes\n", sizeof(long unsigned int) / sizeof(char)); */
    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n_points]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n_points = atol(argv[1]);
    }

    printf("Generating %lu points...\n", n_points);
    const double t_start = omp_get_wtime();
    n_inside = generate_points(n_points);
    const double t_elapsed = omp_get_wtime() - t_start;
    const double pi_approx = 4.0 * n_inside / (double) n_points;
    printf("PI approximation %f, exact %f, error %f%%\n", pi_approx, PI_EXACT, 100.0 * fabs(pi_approx - PI_EXACT) / PI_EXACT);
    printf("!! Elapsed time: %.2f s !!\n", t_elapsed);

    return EXIT_SUCCESS;
}
