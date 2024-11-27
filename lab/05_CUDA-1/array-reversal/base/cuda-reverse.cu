/****************************************************************************
 *
 * cuda-reverse.cu - Array reversal with CUDA
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
#include <math.h>
#include <assert.h>

#include "hpc.h"

/* Reverses `in[]` into `out[]`; assume that `in[]` and `out[]` do not
   overlap.

   [TODO] Modify this function so that it:
   - allocates memory on the device to hold a copy of `in` and `out`;
   - copies `in` to the device
   - launches a kernel (to be defined)
   - copies data back from device to host
   - deallocates memory on the device
 */
void reverse(int *in, int *out, int n)
{
    for (int i = 0; i < n; i++) {
        const int opp = n - 1 - i;
        out[opp] = in[i];
    }
}

/* In-place reversal of in[] into itself.

   [TODO] Modify this function so that it:
   - allocates memory on the device to hold a copy of `in`;
   - copies `in` to the device
   - launches a kernel (to be defined)
   - copies data back from device to host
   - deallocates memory on the device
*/
void inplace_reverse(int *in, int n)
{
    int i = 0, j = n - 1;
    while (i < j) {
        const int tmp = in[j];
        in[j] = in[i];
        in[i] = tmp;
        j--;
        i++;
    }
}

void fill(int *x, int n)
{
    for (int i = 0; i < n; i++) {
        x[i] = i;
    }
}

int check(const int *x, int n)
{
    for (int i = 0; i < n; i++) {
        if (x[i] != n - 1 - i) {
            fprintf(stderr, "Test FAILED: x[%d]=%d, expected %d\n", i, x[i], n - 1 - i);
            return 0;
        }
    }
    printf("Test OK\n");
    return 1;
}

int main(int argc, char *argv[])
{
    int *in, *out;
    int n = 1024 * 1024;
    const int MAX_N = 512 * 1024 * 1024;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (n > MAX_N) {
        fprintf(stderr, "FATAL: input too large (maximum allowed length is %d)\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n * sizeof(*in);

    /* Allocate in[] and out[] */
    in = (int *) malloc(SIZE);
    assert(in != NULL);
    out = (int *) malloc(SIZE);
    assert(out != NULL);
    fill(in, n);

    printf("Reverse %d elements... ", n);
    reverse(in, out, n);
    check(out, n);

    printf("In-place reverse %d elements... ", n);
    inplace_reverse(in, n);
    check(in, n);

    /* Cleanup */
    free(in);
    free(out);

    return EXIT_SUCCESS;
}
