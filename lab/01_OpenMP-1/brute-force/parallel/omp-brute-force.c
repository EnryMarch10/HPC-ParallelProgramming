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

/***
% HPC - Brute-force password cracking
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-09-20

![[DES cracker board](https://en.wikipedia.org/wiki/EFF_DES_cracker) developed in 1998 by the Electronic Frontier Foundation (EFF); this device can be used to brute-force a DES key. The original uploader was Matt Crypto at English Wikipedia. Later versions were uploaded by Ed g2s at en.wikipedia - CC BY 3.0 us, <https://commons.wikimedia.org/w/index.php?curid=2437815>](des-cracker.jpg)

The program [omp-brute-force.c](omp-brute-force.c) contains an
encrypted message stored in the array `enc[]` of length 64.  The
message has been encrypted using the _XOR_ algorithm, which applies
the "exclusive or" (xor) operator between the plaintext and the
encryption key. The _XOR_ algorithm is not secure, unless the key is
truly random and has the same length of the plaintext; however, it is
certainly "good enough" for this exercise.

_XOR_ is a _symmetric_ encryption algorithm, meaning that the same key
must be used for encryption and decryption. The function `xorcrypt(in,
out, n, key, keylen)` is used to encrypt or decrypt a message with a
given key. To encrypt, `in` must point to the plaintext and `out` to a
memory buffer where the ciphertext will be stored. To decrypt, `in`
points to the ciphertext and `out` to a memory buffer where the
plaintext will be stored.

The parameters are as follows:

- `in` points to the source message. This buffer does not need to be
  zero-terminated since it may contain arbitrary bytes.

- `out` points to a memory buffer of at least $n$ Bytes (the same
  length of the source message), that must be allocated by the caller.
  At the end, this buffer contains the source message that has been
  encrypted/decrypted with the encryption key.

- `n` is the length, in Bytes, of the source message.

- `key` points to the encryption/decryption key. The key is a sequence
  of arbitrary bytes, and therefore does not need to be
  zero-terminated.

- `keylen` is the length of the encryption/decryption key.

The _XOR_ algorithm will happily decrypt any message with any key; if
the key is not correct, the decrypted message will not make any
sense.

For this exercise we know that the plaintext is a zero-terminated
ASCII string that begins with `0123456789`. We also know that the
encryption key is a sequence of 8 ASCII numeric characters; therefore,
the key is a string from `"00000000"` and `"99999999"`. The goal is to
write a program to brute-force the key using OpenMP. The program
should try every key until a valid message is found, i.e., a message
that begins with `0123456789`. At the end, the program must print the
plaintext, which is a famous quote from [an old
film](https://en.wikipedia.org/wiki/WarGames).

The main loop can not be parallelized with the `omp for` construct
(why?). Therefore, you must use `omp parallel` and manually partition
the key space among threads. Remember that `omp parallel` can be
applied to a _structured block_, i.e., a block with a single entry and
a single exit point. Therefore, the thread who finds the correct key
can not exit the parallel region using `return`, `break` or `goto`
(the compiler should raise a compile-time error). However, we
certainly want to terminate the program as soon as the key is found.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-brute-force.c -o omp-brute-force

Run with:

        OMP_NUM_THREADS=2 ./omp-brute-force

**Note**: the execution time of the parallel program might change
irregularly depending on $P$. Why?

## Files

You can use `wget` to transfer the files from the Web page to the
server:

        wget https://www.moreno.marzolla.name/teaching/HPC/handouts/omp-brute-force.c

- [omp-brute-force.c](omp-brute-force.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h> /* Va incluso, altrimenti WARNING */

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
    const int msglen = sizeof(enc); /* / sizeof(char) */
    const char check[] = "0123456789"; /* the correctly decrypted message starts with these characters */
    const int CHECK_LEN = strlen(check);
    const int n = 100000000; /* number of possible keys */
    char key[KEY_LEN + 1]; /* sprintf will output the trailing \0, so we need one byte more for the key */
    /* E` corretto anche senza volatile,
       ma alcuni compilatori ignorano le direttive OpenMP.
       Questo forza a dire che la variabile deve essere creata e che e` modificata nel codice.
       Evita bug su certi compialtori. 
       NON E` QUESTO IL CASO!! Era solo per farcelo notare. */
    volatile int found = 0;

    const double t_start = omp_get_wtime();
    /* Non posso usare "omp parallel for" perche`:
       - out deve essere dichiarato in ogni thread
       - found e` usato da tutti i threads
       - il for non e` parallelizzabile per la sua condizione composta
       in questo caso va fatto per forza manualmente */
#pragma omp parallel default(none) shared(check, CHECK_LEN, KEY_LEN, msglen, n, found, enc) private(key)
    {
        // printf("Inside par region: threads=%d, max=%d\n",
        //     omp_get_num_threads(), omp_get_max_threads());
        char * const out = (char*) malloc(msglen); /* where to put the decrypted message */
        assert(out != NULL);
        const int my_id = omp_get_thread_num();
        const int num_threads = omp_get_num_threads();

        const int my_start = (n * my_id) / num_threads;
        const int my_end = (n * (my_id + 1)) / num_threads;

        for (int k = my_start; k < my_end && !found; k++) /* Quella del my_end non e race condition */
        {
            snprintf(key, KEY_LEN + 1, "%08d", k);
            xorcrypt(enc, out, msglen, key, KEY_LEN); /* Cifro e DECIFRO (qui) con la stessa funzione */
            /* `out` contains the decrypted text; if the key is not
                correct, `out` will contain garbage */
            if (memcmp(out, check, CHECK_LEN) == 0) {
                printf("Key found: %s\n", key);
                printf("Decrypted message: \"%s\"\n", out);
                found = 1;
            }
        }
        free(out);
    }
    const double t_end = omp_get_wtime();
    printf("!! Elapsed time: %.2f s !!\n", t_end - t_start);
    /* Lo speed-up non aumenta linearmente all'aumentare il numero dei threads con:
       > OMP_NUM_THREADS=N ./omp-brute-force
       con N crescente.
       Questo perche` il numero dei threads determina dove viene partizionata l'analisi delle chiavi.
       Il che` puo` essere piu` o meno vantaggioso nel caso di piu` o meno threads.
       Quindi in questo specifico caso non e` detto che all'aumentare del numero di threads il tempo cali.
       Nonostante questo la versione parallela e` molto piu` veloce di quella seriale (per questo stesso motivo). */
    assert(found); /* ensure that we did found the key */
    return EXIT_SUCCESS;
}
