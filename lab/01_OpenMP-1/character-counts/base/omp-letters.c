/****************************************************************************
 *
 * omp-letters.c - Character counts
 *
 * Copyright (C) 2018--2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 ****************************************************************************/

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
    int nlet = 0; /* total number of alphabetic characters processed */
    const size_t TEXT_LEN = strlen(text);
    /* [TODO] Parallelize this function */

    /* Reset the histogram */
    for (int j = 0; j < ALPHA_SIZE; j++) {
        hist[j] = 0;
    }

    /* Count occurrences */
    for (int i = 0; i < TEXT_LEN; i++) {
        const char c = text[i];
        if (isalpha(c)) {
            nlet++;
            hist[tolower(c) - 'a']++;
        }
    }

    return nlet;
}

/**
 * If `freq == 100`, draw `len` caracters; otherwise, draw a fraction
 * of `len` characters proportional to `freq`.
 */
void bar(float freq, int len)
{
    for (int i = 0; i < len * freq / 100; i++) {
        printf("#");
    }
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
        const float freq = 100.0 * hist[i] / nlet;
        printf("%c : %8d (%6.2f%%) ", 'a' + i, hist[i], freq);
        bar(freq, 65);
        printf("\n");
    }
    printf("    %8d total\n", nlet);
}

int main(void)
{
    int hist[ALPHA_SIZE];
    const size_t size = 5 * 1024 * 1024; /* maximum text size: 5 MB */
    char *text = (char *) malloc(size);
    assert(text != NULL);

    const size_t len = fread(text, 1, size - 1, stdin);
    text[len] = '\0'; /* put a termination mark at the end of the text */
    const double tstart = omp_get_wtime();
    make_hist(text, hist);
    const double elapsed = omp_get_wtime() - tstart;
    print_hist(hist);
    fprintf(stderr, "Elapsed time: %f\n", elapsed);
    free(text);
    return EXIT_SUCCESS;
}
