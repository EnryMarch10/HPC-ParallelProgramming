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
    const int msglen = sizeof(enc);
    const char check[] = "0123456789"; /* the correctly decrypted message starts with these characters */
    const int CHECK_LEN = strlen(check);
    const int n = 100000000; /* number of possible keys */
    char key[KEY_LEN + 1]; /* sprintf will output the trailing \0, so we need one byte more for the key */
    int k; /* numeric value of the key to try */
    volatile int found = 0;
    char* out = (char*) malloc(msglen); /* where to put the decrypted message */
    assert(out != NULL);

    for (k = 0; k < n && !found; k++) {
        snprintf(key, KEY_LEN + 1, "%08d", k);
        xorcrypt(enc, out, msglen, key, KEY_LEN);
        /* `out` contains the decrypted text; if the key is not
           correct, `out` will contain garbage */
        if (memcmp(out, check, CHECK_LEN) == 0) {
            printf("Key found: %s\n", key);
            printf("Decrypted message: \"%s\"\n", out);
            found = 1;
        }
    }
    assert(found); /* ensure that we did found the key */
    free(out);
    return EXIT_SUCCESS;
}
