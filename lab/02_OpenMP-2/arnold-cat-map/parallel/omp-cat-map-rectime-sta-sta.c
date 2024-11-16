#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <limits.h>

#define MAX_IT 5000

/* Compute the Greatest Common Divisor (GCD) of integers a > 0 and b > 0 using the Euclidean algorithm */
int gcd(int a, int b)
{
    assert(a > 0);
    assert(b > 0);

    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/* Compute the Least Common Multiple (LCM) of integers a > 0 and b > 0 */
int lcm(int a, int b)
{
    assert(a > 0);
    assert(b > 0);
    return (a / gcd(a, b)) * b;
}

/**
 * Compute the recurrence time of Arnold's cat map applied to an image
 * of size (n * n). For each point (x,y), compute the minimum recurrence
 * time k(x,y). The minimum recurrence time for the whole image is the
 * Least Common Multiple of all k(x,y).
 */
int cat_map_rectime(int n)
{
    assert(n > 0);
    const int it_size = n * n;
    int *it = (int *) malloc(it_size * sizeof(int));

#pragma omp parallel for collapse(2) default(none) shared(n, it)
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
            int x_cur = x, y_cur = y;
            int i;
            // TODO: gestire eccezione del numero di iterazioni troppo grande
            for (i = 0; i < MAX_IT; i++) {
                const int x_next = (2 * x_cur + y_cur) % n;
                const int y_next = (x_cur + y_cur) % n;
                x_cur = x_next;
                y_cur = y_next;
                if (x_cur == x && y_cur == y) {
                    break;
                }
            }
            it[x + y * n] = i + 1;
        }
    }
    const int max_threads = omp_get_max_threads();
    int *new_it = (int *) malloc(max_threads * sizeof(int));
#pragma omp parallel num_threads(max_threads) default(none) shared(max_threads, new_it, it, it_size)
    {
        const int my_id = omp_get_thread_num();
        new_it[my_id] = 1;

        const int my_start = (it_size * my_id) / max_threads;
        const int my_end = (it_size * (my_id + 1)) / max_threads;

        for (int i = my_start; i < my_end; i++) {
            new_it[my_id] = lcm(new_it[my_id], it[i]);
        }
    }
    free(it);
    int min_it = 1;
    for (int i = 0; i < max_threads; i++) {
        min_it = lcm(min_it, new_it[i]);
    }
    free(new_it);
    return min_it;
}

int main(int argc, char* argv[])
{
    int n, k;
    if (argc != 2) {
        fprintf(stderr, "Usage: %s image_size\n", argv[0]);
        return EXIT_FAILURE;
    }
    n = atoi(argv[1]);
    const double t_start = omp_get_wtime();
    k = cat_map_rectime(n);
    const double t_elapsed = omp_get_wtime() - t_start;
    printf("%d\n", k);

    printf("!! Elapsed time: %.2f s !!\n", t_elapsed);

    return EXIT_SUCCESS;
}
