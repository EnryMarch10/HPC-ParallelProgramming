#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

void fill(int *v1, int *v2, size_t n)
{
    const int seq1[3] = { 3, 7, 18 };
    const int seq2[3] = { 12, 0, -2 };
#pragma omp parallel for default(none) shared(v1, v2, n, seq1, seq2) /* Non richiesto, ma velocizza l'esecuzione */
    for (size_t i = 0; i <  n; i++) {
        v1[i] = seq1[i%3];
        v2[i] = seq2[i%3];
    }
}

int dot(const int *v1, const int *v2, size_t n)
{
    int result = 0;
#pragma omp parallel for default(none) shared(v1, v2, n) reduction(+:result)
    for (size_t i = 0; i < n; i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

int main(int argc, char *argv[])
{
    size_t n = 10 * 1024 * 1024l; /* array length */
    const size_t n_max = 512 * 1024 * 1024l; /* max length */
    int *v1, *v2;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atol(argv[1]);
    }

    if (n > n_max) {
        fprintf(stderr, "FATAL: Array too long (requested length=%lu, maximum length=%lu\n", (unsigned long) n, (unsigned long) n_max);
        return EXIT_FAILURE;
    }

    printf("Initializing array of length %lu\n", (unsigned long) n);
    v1 = (int *) malloc(n * sizeof(v1[0]));
    assert(v1 != NULL);
    v2 = (int *) malloc(n * sizeof(v2[0]));
    assert(v2 != NULL);
    fill(v1, v2, n);

    const int expect = n % 3 == 0 ? 0 : 36;

    const double t_start = omp_get_wtime();
    const int result = dot(v1, v2, n);
    const double t_end = omp_get_wtime();

    if (result == expect) {
        printf("Test OK\n");
    } else {
        printf("Test FAILED: expected %d, got %d\n", expect, result);
    }
    printf("!! Elapsed time: %.2f s !!\n", t_end - t_start);
    free(v1);
    free(v2);

    return EXIT_SUCCESS;
}
