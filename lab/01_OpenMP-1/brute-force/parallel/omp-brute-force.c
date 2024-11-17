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
        char *const out = (char *) malloc(msglen); /* where to put the decrypted message */
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
