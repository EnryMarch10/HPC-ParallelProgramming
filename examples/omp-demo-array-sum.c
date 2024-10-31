/******************************************************************************

Sequential and parallel solutions (some of which are WRONG) to the
problem of summing the elements in an array.

To compile with gcc:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-demo-array-sum.c -o omp-demo-array-sum

To execute:

        ./arraySum 20000

(the optional parameter is the array length).

To specify the number of OpenMP threads:

        OMP_NUM_THREADS=6 ./arraySum 20000

Original author: Gianluigi Zavattaro, Università di Bologna
Last modified on 2024-09-15 by Moreno Marzolla, Università di Bologna

******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <omp.h>

// sequential
long sum1(const int* A, int n)
{
    long global_sum = 0;
    for (int i = 0; i < n; i++) {
        global_sum += A[i];
    }
    return global_sum;
}

// parallel but WRONG: concurrent updates of the counter
long sum2(const int* A, int n)
{
    long global_sum = 0;
#pragma omp parallel
    {
        const int my_id = omp_get_thread_num();
        const int my_num_threads = omp_get_num_threads();
        const int my_block_len = n / my_num_threads;
        const int my_start = my_id * my_block_len;
        const int my_end = my_start + my_block_len;
        for (int my_i = my_start; my_i < my_end; my_i++) {
            global_sum += A[my_i];
        }
    }
    return global_sum;
}

// parallel with mutex on counter but WRONG partitioning (if
// my_num_threads is not a divisor of size)
long sum3(const int* A, int n)
{
    long global_sum = 0;
#pragma omp parallel
    {
        const int my_id = omp_get_thread_num();
        const int my_num_threads = omp_get_num_threads();
        const int my_block_len = n / my_num_threads;
        const int my_start = my_id * my_block_len;
        const int my_end = my_start + my_block_len;
        for (int my_i = my_start; my_i < my_end; my_i++) {
#pragma omp atomic
            global_sum += A[my_i];
        }
    }
    return global_sum;
}

// parallel with mutex on counter
long sum4(const int* A, int n)
{
    long global_sum = 0;
#pragma omp parallel
    {
        const int my_id = omp_get_thread_num();
        const int my_num_threads = omp_get_num_threads();
        const int my_start = n * my_id / my_num_threads;
        const int my_end = n * (my_id + 1) / my_num_threads;
        for (int my_i = my_start; my_i < my_end; my_i++) {
#pragma omp atomic
            global_sum += A[my_i];
        }
    }
    return global_sum;
}

// parallel with local counters and mutex on the global counter
long sum5(const int* A, int n)
{
    long global_sum = 0;
#pragma omp parallel
    {
        const int my_id = omp_get_thread_num();
        const int my_num_threads = omp_get_num_threads();
        const int my_start = n * my_id / my_num_threads;
        const int my_end = n * (my_id + 1) / my_num_threads;
        long my_sum = 0;
        for (int my_i = my_start; my_i < my_end; my_i++) {
            my_sum += A[my_i];
        }
#pragma omp atomic
        global_sum += my_sum;
    }
    return global_sum;
}

// parallel without mutex - WRONG without barrier synchronization
long sum6(const int* A, int n)
{
    int num_threads = omp_get_max_threads();
    long psum[num_threads];
    long global_sum = 0;
#pragma omp parallel
    {
        const int my_id = omp_get_thread_num();
        const int my_start = n * my_id / num_threads;
        const int my_end = n * (my_id + 1) / num_threads;
        long my_sum = 0;
        for (int my_i = my_start; my_i < my_end; my_i++) {
            my_sum += A[my_i];
        }
        psum[my_id] = my_sum;
        if (my_id == 0) {
            for (int my_i = 0; my_i < num_threads; my_i++)
                global_sum += psum[my_i];
        }
    }
    return global_sum;
}

// parallel without mutex and with barrier synchronization
long sum7(const int* A, int n)
{
    int num_threads = omp_get_max_threads();
    long psum[num_threads];
    long global_sum = 0;
#pragma omp parallel
    {
        const int my_id = omp_get_thread_num();
        const int my_start = n * my_id / num_threads;
        const int my_end = n * (my_id + 1) / num_threads;
        long my_sum = 0;
        for (int my_i = my_start; my_i < my_end; my_i++) {
            my_sum += A[my_i];
        }
        psum[my_id] = my_sum;
    } // implicit synchronization at end of parallel region
    for (int i = 0; i < num_threads; i++)
        global_sum += psum[i];
    return global_sum;
}

// parallel with parallel for and automatic reduction
long sum8(const int* A, int n)
{
    long global_sum = 0;
#pragma omp parallel for reduction(+:global_sum)
    for (int i = 0; i < n; i++) {
        global_sum += A[i];
    }
    return global_sum;
}

void test_func( const char *desc, long (* func)(const int*, int), const int *A, int n, long expect )
{
    printf("*** %s\n", desc);
    const long result = func(A, n);
    printf("Sum = %ld (%s)\n", result, result == expect ? "OK" : "FAIL");
}

int main(int argc, char *argv[])
{
    int len = 1001;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [len]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        len = atoi(argv[1]);
    }

    int *A = (int*)malloc(len * sizeof(*A));
    assert(A != NULL);

    srand((unsigned int)time(NULL));
    for (int i = 0; i < len; i++)
    	A[i] = rand()%2;

    const long sequential_sum = sum1(A, len);

    test_func("SEQUENTIAL",
              sum1, A, len, sequential_sum);

    test_func("PARALLEL WITH CONCURRENT UPDATES OF COUNTER",
              sum2, A, len, sequential_sum);

    test_func("PARALLEL WITH MUTEX BUT WRONG PARTITIONING",
              sum3, A, len, sequential_sum);

    test_func("PARALLEL WITH MUTEX",
              sum4, A, len, sequential_sum);

    test_func("PARALLEL WITH LOCAL COUNTERS AND MUTEX ON GLOBAL COUNTER",
              sum5, A, len, sequential_sum);

    test_func("PARALLEL WITH NO MUTEX BUT MISSING BARRIER",
              sum6, A, len, sequential_sum);

    test_func("PARALLEL WITH NO MUTEX WITH BARRIER",
              sum7, A, len, sequential_sum);

    test_func("PARALLEL USING PARALLEL FOR AND REDUCTION",
              sum8, A, len, sequential_sum);

    free(A);
    return EXIT_SUCCESS;
}
