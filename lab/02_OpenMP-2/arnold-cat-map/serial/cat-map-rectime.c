#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <time.h>

#define MAX_IT 5000

/* Compute the Greatest Common Divisor (GCD) of integers a > 0 and b > 0 */
/* int gcd(int a, int b)
{
    assert(a > 0);
    assert(b > 0);

    while (b != a) {
        if (a > b) {
            a = a - b;
        } else {
            b = b - a;
        }
    }
    return a;
} */

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
    int it = 1;
    /* [TODO] Implement this function; start with a working serial
       version, then parallelize. */
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
            it = lcm(it, i + 1);
        }
    }
    return it;
}

int main(int argc, char* argv[])
{
    int n, k;
    if (argc != 2) {
        fprintf(stderr, "Usage: %s image_size\n", argv[0]);
        return EXIT_FAILURE;
    }
    n = atoi(argv[1]);
    const clock_t t_start = clock();
    k = cat_map_rectime(n);
    const clock_t t_end = clock();
    printf("n iterations required = %d\n", k);

    printf("!! Elapsed time: %.2f s !!\n", ((double) (t_end - t_start) / CLOCKS_PER_SEC));

    return EXIT_SUCCESS;
}
