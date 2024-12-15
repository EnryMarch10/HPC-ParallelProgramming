/******************************************************************************
 *
 * omp-levenshtein.c - Levenshtein's edit distance
 *
 * Written in 2017--2022, 2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

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

    /* [TODO] Parallelize this */
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
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
