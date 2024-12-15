#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

int min3(int a, int b, int c)
{
    const int minab = a < b ? a : b;
    return c < minab ? c : minab;
}

/* This function computes the Levenshtein edit distance between
   strings s and t. If we let n = strlen(s) and m = strlen(t), this
   function uses time O(nm) and space O(nm). */
int levenshtein(const char *s, const char *t)
{
    const int n = strlen(s), m = strlen(t);
    int (*L)[m + 1] = malloc((n + 1) * (m + 1) * sizeof(int)); /* C99 idiom: L is of type int L[][m+1] */
    int result;

    /* degenerate cases first */
    if (n == 0) {
        return m;
    }
    if (m == 0) {
        return n;
    }

    /* Initialize the first column of L */
    for (int i = 0; i <= n; i++) {
        L[i][0] = i;
    }

    /* Initialize the first row of L */
    for (int j = 0; j <= m; j++) {
        L[0][j] = j;
    }

    for (int slice = 0; slice < n + m - 1; slice++) {
        const int z1 = slice < m ? 0 : slice - m + 1;
        const int z2 = slice < n ? 0 : slice - n + 1;
#pragma omp parallel for default(none) shared(slice, L, s, t, z1, z2, m)
        for (int ii = slice - z2; ii >= z1; ii--) {
            const int jj = slice - ii;
            const int i = ii + 1;
            const int j = jj + 1;
            L[i][j] = min3(L[i - 1][j] + 1,
                           L[i][j - 1] + 1,
                           L[i - 1][j - 1] + (s[i - 1] != t[j - 1]));
        }
    }
    result = L[n][m];
    free(L);
    return result;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s str1 str2\n", argv[0]);
        return EXIT_FAILURE;
    }

    printf("%d\n", levenshtein(argv[1], argv[2]));
    return EXIT_SUCCESS;
}
