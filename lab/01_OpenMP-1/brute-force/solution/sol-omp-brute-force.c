/****************************************************************************
 *
 * omp-brute-force.c - Brute-force password cracking
 *
 * Copyright (C) 2017--2022, 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
#include <string.h>
#include <assert.h>
#include <omp.h>

/* Decrypt `enc` of length `n` bytes into buffer `dec` using `key` of
   length `keylen`. The encrypted message, decrypted messages and key
   are treated as binary blobs; hence, they do not need to be
   zero-terminated.

   Do not modify this function. */
void xorcrypt(const char* in, char* out, int n, const char* key, int keylen)
{
    for (int i = 0; i < n; i++) {
        out[i] = in[i] ^ key[i % keylen];
    }
}

/* Encrypt message `msg` using key `key` of length `keylen`. `mst`
   must be a zero-terminated ASCII string. Returns a pointer to a
   newly allocated block of length `(strlen(msg)+1)` containing the
   encrypted message. */
char *gen_encrypt(const char *msg, char *key, int keylen)
{
    const int n = strlen(msg) + 1;
    char* out = malloc(n);
    int i;

    assert(out != NULL);
    xorcrypt(msg, out, n, key, keylen);
    printf("const char enc[] = {");
    for (i = 0; i < n; i++) {
        if (i % 8 == 0) {
            printf("\n");
        }
        printf("%d", out[i]);
        if (i < n - 1) {
            printf(", ");
        }
    }
    printf("\n};\n");
    return out;
}

int main(int argc, char *argv[])
{
    const int KEY_LEN = 8;
    /* encrypted message */
    const char enc[] = {
        4, 1, 0, 1, 0, 1, 4, 1,
        12, 9, 115, 18, 71, 64, 64, 87,
        90, 87, 87, 18, 83, 85, 95, 83,
        26, 16, 102, 90, 81, 20, 93, 88,
        88, 73, 18, 69, 93, 90, 92, 95,
        90, 87, 18, 95, 91, 66, 87, 22,
        93, 67, 18, 92, 91, 64, 18, 66,
        91, 16, 66, 94, 85, 77, 28, 54
    };
    /* There is some redundant code that has been used by me to
       generate the encrypted message */
    const char *msg = "0123456789A strange game. The only winning move is not to play."; /* plaintext message */
    const int msglen = strlen(msg) + 1; /* length of the encrypted message, including the trailing \0 */
    char enc_key[] = "40224426"; /* encryption key */
    const int n = 100000000; /* total number of possible keys */
    volatile int found = 0;
    const char check[] = "0123456789"; /* the decrypted message starts with this string */
    const int CHECK_LEN = strlen(check);

    char *tmp = gen_encrypt(msg, enc_key, KEY_LEN);
    free(tmp);

    const double tstart = omp_get_wtime();
#pragma omp parallel default(none) shared(found, check, enc, msglen, n, CHECK_LEN, KEY_LEN)
    {
        char* out = (char *) malloc(msglen);
        char key[KEY_LEN + 1];
        const int my_id = omp_get_thread_num();
        const int num_threads = omp_get_num_threads();
        const int my_start = (n * my_id) / num_threads;
        const int my_end = (n * (my_id + 1)) / num_threads;
        /* Technically, there is a race condition updating the
           variable `found`; however, the race condition is benign
           because in the worst case it forces the other threads to
           execute one more iteration than necessary. */
        for (int k = my_start; k < my_end && !found; k++) {
            sprintf(key, "%08d", k);
            xorcrypt(enc, out, msglen, key, 8);
            if (memcmp(out, check, CHECK_LEN) == 0) {
                printf("Key found: %s\n", key);
                printf("Decrypted message: \"%s\"\n", out);
                found = 1;
            }
        }
        free(out);
    }
    const double elapsed = omp_get_wtime() - tstart;
    printf("Elapsed time: %f\n", elapsed);
    return EXIT_SUCCESS;
}
