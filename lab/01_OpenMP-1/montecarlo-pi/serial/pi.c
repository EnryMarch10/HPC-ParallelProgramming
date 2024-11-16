/* The rand_r() function is available only if _XOPEN_SOURCE=600 */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs */
#include <time.h>

/* Generate `n` random points within the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1. */
unsigned int generate_points(unsigned int n)
{
    /* [TODO] parallelize the body of this function */
    unsigned int n_inside = 0;
    /* The C function `rand()` is not thread-safe, since it modifies a
       global seed; therefore, it can not be used inside a parallel
       region. We use `rand_r()` with an explicit per-thread
       seed. However, this means that in general the result computed
       by this program depends on the number of threads. */
    unsigned int my_seed = 17 + 19 * 0 /* omp_get_thread_num() */;
    for (int i = 0; i < n; i++) {
        /* Generate two random values in the range [-1, 1] */
        const double x = (2.0 * rand_r(&my_seed) / (double) RAND_MAX) - 1.0;
        const double y = (2.0 * rand_r(&my_seed) / (double) RAND_MAX) - 1.0;
        if (x * x + y * y <= 1.0) {
            n_inside++;
        }
    }
    return n_inside;
}

int main(int argc, char *argv[])
{
    unsigned int n_points = 10000;
    unsigned int n_inside;
    const double PI_EXACT = 3.14159265358979323846;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n_points]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n_points = atol(argv[1]);
    }

    printf("Generating %u points...\n", n_points);
    const clock_t t_start = clock();
    n_inside = generate_points(n_points);
    const clock_t t_end = clock();
    const double pi_approx = 4.0 * n_inside / (double) n_points;
    printf("PI approximation %f, exact %f, error %f%%\n", pi_approx, PI_EXACT, 100.0 * fabs(pi_approx - PI_EXACT) / PI_EXACT);
    printf("!! Elapsed time: %.2f s !!\n", ((double) (t_end - t_start) / CLOCKS_PER_SEC));

    return EXIT_SUCCESS;
}
