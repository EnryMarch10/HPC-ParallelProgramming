#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

/* Recursive computation of the n-th Fibonacci number, for n=0, 1, 2, ...
   Do not parallelize this function. */
int fib_rec(int n)
{
    if (n < 2) {
        return 1;
    } else {
        return fib_rec(n - 1) + fib_rec(n - 2);
    }
}

/* Iterative computation of the n-th Fibonacci number. This function
   must be used for checking the result only. */
int fib_iter(int n)
{
    if (n < 2) {
        return 1;
    } else {
        int fibnm1 = 1;
        int fibnm2 = 1;
        int fibn;
        n--;
        do {
            fibn = fibnm1 + fibnm2;
            fibnm2 = fibnm1;
            fibnm1 = fibn;
            n--;
        } while (n > 0);
        return fibn;
    }
}

/* Fill vectors `vin` and `vout` of length `n`; `vin` will contain
   input values; `vout` is initialized with -1 */
void fill(int *vin, int *vout, int n)
{
    for (int i = 0; i < n; i++) {
        vin[i] = 25 + (i % 10);
        vout[i] = -1;
    }
}

/* Check correctness of `vout[]`. Return 1 if correct, 0 if not */
int is_correct(const int *vin, const int *vout, int n)
{
    for (int i = 0; i < n; i++) {
        if (vout[i] != fib_iter(vin[i]) ) {
            fprintf(stderr,
                    "Test FAILED: vin[%d]=%d, vout[%d]=%d (expected %d)\n",
                    i, vin[i], i, vout[i], fib_iter(vin[i]));
            return 0;
        }
    }
    fprintf(stderr, "Test OK\n");
    return 1;
}


void do_static(const int *vin, int *vout, const int n, const int chunk_size)
{
    /* [TODO] parallelize the following loop, simulating a
       "schedule(static, 1)" clause, i.e., static scheduling with block
       size 1. Optionally, allow an arbitrary chunk size.

       Hint: the iteration space i=0..n-1 should be partitioned into
       blocks of length `chunk_size`.  the blocks are assigned to
       OpenMP threads using a cyclic assignment, e.g.:

        <---------- STRIDE --------->
        +----------+----------+-----+----------+----------+-----
       0|    P0    |    P1    | ... |    P0    |    P1    | ...
        +----------+----------+-----+----------+----------+-----
         chunk_size chunk_size       chunk_size chunk_size

       Let `STRIDE` be the number of iterations between the beginning
       of a chunk assigned to thread `p` and the next chunk assigned
       to the same thread. Therefore:

       STRIDE = num_threads * chunk_size;

       The first chunk assigned to `p` starts at `(p * chunk_size)`.
       Therefore, each thread should execute the following nested
       loops:

       START = my_id * chunk_size;
       STRIDE = num_threads * chunk_size;
       for (int i=START; i<n; i+=STRIDE) {
         for (int j=i; j<i+chunk_size && j<n; j++) {
           loop body
         }
       }

       Note that `n` is not necessarily an integer multiple of the
       number of threads, therefore addtional checks are necessary to
       ensure that we never exceed the upper bound `n-1`.
    */
#pragma omp parallel default(none) shared(vin, vout, n, chunk_size)
    {
        const int stride = omp_get_num_threads() * chunk_size;
        const int my_id = omp_get_thread_num();

        for (int i = my_id * chunk_size; i < n; i += stride) {
            for (int j = i; j < i + chunk_size && j < n; j++) {
                vout[j] = fib_rec(vin[j]);
            }
        }
    }
}

void do_dynamic(const int *vin, int *vout, const int n, const int chunk_size)
{
    /* [TODO] parallelize the following loop, simulating a
       "schedule(dynamic,1)" clause, i.e., dynamic scheduling with
       chunk size 1. Optionally, allow the user to specify the chunk
       size.

       Hint: keep a shared variable `idx` representing the index of
       the beginning of the first unprocessed chunk, i.e., the first
       chunk that will be assigned to a thread.

       Each OpenMP thread _atomically_ fetches the current value of
       `idx` into a local (private) variable `my_idx`, and then
       increments `idx` by `chunk_size`.

       Therefore, each thread executes the following pseudocode:

       do {
         atomically copy idx into my_idx and increment idx by chunk_size
         for (i=my_idx; i<my_idx + chunk_size && i<n; i++) {
           loop body
         }
       } while (my_idx < n);
    */
    int index = 0;
#pragma omp parallel default(none) shared(vin, vout, n, chunk_size, index)
    {
        int my_index;
        do
        {
            // atomically copy idx into my_idx and increment idx by chunk_size
#pragma omp critical
            {
                my_index = index;
                index += chunk_size;
            }
            for (int i = my_index; i < my_index + chunk_size && i < n; i++) {
                vout[i] = fib_rec(vin[i]);
            }
        } while (my_index < n);
    }
}

int main(int argc, char* argv[])
{
    int n = 1024;
    int chunk_size = 1;
    const int max_n = 512 * 1024 * 1024;
    int *vin, *vout;
    double t_start, t_elapsed;

    if (argc > 3) {
        fprintf(stderr, "Usage: %s [n] [chunk_size]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 3) {
        n = atoi(argv[1]);
        chunk_size = atoi(argv[2]);
    }
    else if (argc == 2) {
        n = atoi(argv[1]);
    }

    if (n > max_n) {
        fprintf(stderr, "FATAL: n too large (max value is %d)\n", max_n);
        return EXIT_FAILURE;
    }

    /* initialize the input and output arrays */
    vin = (int *) malloc(n * sizeof(vin[0]));
    assert(vin != NULL);
    vout = (int *) malloc(n * sizeof(vout[0]));
    assert(vout != NULL);

    /**
     ** Test static schedule implementation
     **/
    fill(vin, vout, n);
    t_start = omp_get_wtime();
    do_static(vin, vout, n, chunk_size);
    t_elapsed = omp_get_wtime() - t_start;
    is_correct(vin, vout, n);

    printf("!! Elapsed time (static schedule): %.2f s !!\n", t_elapsed);

    /**
     ** Test dynamic schedule implementation
     **/
    fill(vin, vout, n);
    t_start = omp_get_wtime();
    do_dynamic(vin, vout, n, chunk_size);
    t_elapsed = omp_get_wtime() - t_start;
    is_correct(vin, vout, n);

    printf("!! Elapsed time (dynamic schedule): %.2f s !!\n", t_elapsed);

    free(vin);
    free(vout);
    return EXIT_SUCCESS;
}
