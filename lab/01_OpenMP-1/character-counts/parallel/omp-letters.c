#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#define ALPHA_SIZE 26

/**
 * Count occurrences of letters 'a'..'z' in `text`; uppercase
 * characters are transformed into lowercase, and all other symbols
 * are ignored. `text` must be zero-terminated. `hist` will be filled
 * with the computed counts. Returns the total number of letters
 * found.
 */
int make_hist(const char *text, int hist[ALPHA_SIZE])
{
    int n_let = 0; /* total number of alphabetic characters processed */
    const size_t TEXT_LEN = strlen(text);

    /* Reset the histogram */
    for (int j = 0; j < ALPHA_SIZE; j++) {
        hist[j] = 0;
    }

    /* Count occurrences */
#pragma omp parallel for default(none) shared(text, TEXT_LEN) reduction(+:n_let) reduction(+:hist[:ALPHA_SIZE])
    for (int i = 0; i < TEXT_LEN; i++) {
        const char c = text[i];
        if (isalpha(c)) {
            n_let++;
            hist[tolower(c) - 'a']++;
        }
    }

    return n_let;
}

/**
 * Print frequencies
 */
void print_hist(int hist[ALPHA_SIZE])
{
    int nlet = 0;
    for (int i = 0; i < ALPHA_SIZE; i++) {
        nlet += hist[i];
    }
    for (int i = 0; i < ALPHA_SIZE; i++) {
        printf("%c : %8d (%6.2f%%)\n", 'a' + i, hist[i], 100.0 * hist[i] / nlet);
    }
    printf("    %8d total\n", nlet);
}

int main(void)
{
    int hist[ALPHA_SIZE];
    const size_t size = 5 * 1024 * 1024; /* maximum text size: 5 MB */
    char *text = (char*) malloc(size);
    assert(text != NULL);

    const size_t len = fread(text, 1, size - 1, stdin);
    text[len] = '\0'; /* put a termination mark at the end of the text */
    const double t_start = omp_get_wtime();
    make_hist(text, hist);
    const double elapsed = omp_get_wtime() - t_start;
    print_hist(hist);
    fprintf(stderr, "!! Elapsed time: %.5f s !!\n", elapsed);
    free(text);
    return EXIT_SUCCESS;
}
