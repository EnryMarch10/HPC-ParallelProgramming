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

    int exit_flag = 0;
#pragma omp parallel for collapse(2) default(none) shared(n, it, exit_flag)
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
            int x_cur = x, y_cur = y;
            int i;
            for (i = 0; i < MAX_IT; i++) {
                const int x_next = (2 * x_cur + y_cur) % n;
                const int y_next = (x_cur + y_cur) % n;
                x_cur = x_next;
                y_cur = y_next;
                if (x_cur == x && y_cur == y) {
                    break;
                }
            }
            if (i == MAX_IT - 1 && (x_cur != x || y_cur != y)) {
#pragma omp critical // atomic => used before but gives ERROR!
                exit_flag = 1;
            }
            it[x + y * n] = i + 1;
        }
    }
    if (exit_flag) {
        fprintf(stderr, "Overrun exception, evaluation of single point exceeded %d iterations\n", MAX_IT);
        exit(EXIT_FAILURE);
    }
    /* 1 is the neutral element of the Least Common Multiple computation */
    int min_it = 1;
    for (int i = 0; i < it_size; i++) {
        min_it = lcm(min_it, it[i]);
    }
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
