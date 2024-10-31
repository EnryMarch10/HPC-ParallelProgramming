/****************************************************************************
 *
 * omp-merge-sort.c - Merge Sort with OpenMP tasks
 *
 * Copyright (C) 2017--2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Merge Sort with OpenMP tasks
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-10-23

The file [omp-merge-sort.c](omp-merge-sort.c) contains a recursive
implementation of the _Merge Sort_ algorithm. The program uses
_Selection Sort_ when the size of the subvector is less than a
user-defined cutoff value; this is a standard optimization that avoids
the overhead of recursive calls on small vectors.

The program generates and sorts a random permutation of $0, 1, \ldots,
n-1$; it if therefore easy to check the correctness of the result,
since it must be the sequence $0, 1, \ldots, n-1$.

The goal is to parallelize the Merge Sort algorithm using OpenMP
tasks as follows:

- The recursion starts inside a parallel region; only one process
  starts the recursion.

- Create two tasks for the two recursive calls; pay attention to the
  visibility (scope) of variables.

- Wait for the two sub-tasks to complete before starting the _merge_
  step.

Measure the execution time of the parallel program and compare it with
the serial implementation. To get meaningful results, choose an input
size that requires at least a few seconds to be sorted using all
available cores.

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-merge-sort.c -o omp-merge-sort

To execute:

        ./omp-merge-sort 50000

## Files

- [omp-merge-sort.c](omp-merge-sort.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
// #include <time.h>

#define MIN -9999999
#define MAX 99999999
#define PRINT_RIGHT_ALIGN(file, val) fprintf(file, "%8d", (val))

#define FILE_UNSORTED "unsorted.txt"
#define FILE_SORTED "sorted.txt"

int min(int a, int b)
{
    return (a < b ? a : b);
}

void swap(int* a, int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

/**
 * Sort v[low..high] using selection sort. This function will be used
 * for small vectors only. Do not parallelize this.
 */
void selectionsort(int* v, int low, int high)
{
    for (int i = low; i < high; i++) {
        for (int j = i + 1; j <= high; j++) {
            if (v[i] > v[j]) {
                swap(&v[i], &v[j]);
            }
        }
    }
}

/**
 * Merge src[low..mid] with src[mid+1..high], put the result in
 * dst[low..high].
 *
 * Do not parallelize this function (it could be done, but is very
 * difficult, see
 * http://www.drdobbs.com/parallel/parallel-merge/229204454
 * https://en.wikipedia.org/wiki/Merge_algorithm#Parallel_merge )
 */
void merge(int* src, int low, int mid, int high, int* dst)
{
    int i = low, j = mid + 1, k = low;
    while (i <= mid && j <= high) {
        if (src[i] <= src[j]) {
            dst[k] = src[i++];
        } else {
            dst[k] = src[j++];
        }
        k++;
    }
    /* Handle leftovers */
    while (i <= mid) {
        dst[k] = src[i++];
        k++;
    }
    while (j <= high) {
        dst[k] = src[j++];
        k++;
    }
}

/**
 * Sort `v[i..j]` using the recursive version of Merge Sort; the array
 * `tmp[i..j]` is used as a temporary buffer; the caller is
 * responsible for providing a suitably sized array `tmp`. This
 * function must not free `tmp`.
 */
void mergesort_rec(int* v, int i, int j, int* tmp)
{
    const int CUTOFF = 64;
    /* If the sub-vector is smaller than CUTOFF, use selection
       sort. This is a widely used optimization that avoids the
       overhead of recursion for small vectors. The optimal CUTOFF
       value is implementation-dependent; the value used here is just
       an example. */
    if (j - i + 1 < CUTOFF)
        selectionsort(v, i, j);
    else {
        const int m = (i + j) / 2;
        /* [TODO] The two recursive invocations of `mergesort_rec()`
           are independent and can be executed in parallel. Create two
           OpenMP tasks, and wait for their completion before merging
           the results. Pay attention to the visibility of variables
           associated to the tasks.

           `v`, `i`, `m`, `tmp` are local variables, so they are
           `firstprivate` by default according to the visibility rules
           for tasks. However, due to the `taskwait` directive below,
           the values of these variables can not change between task
           creation and execution, so they can be made all
           `shared`. */
#pragma omp task shared(v, i, m, tmp)
        mergesort_rec(v, i, m, tmp);
#pragma omp task shared(v, j, m, tmp)
        mergesort_rec(v, m + 1, j, tmp);
#pragma omp taskwait
        /* Wait for completion of the recursive invocations of
           `mergesort_rec()` before merging. */
        merge(v, i, m, j, tmp);
        /* copy the sorted data back to v */
        memcpy(v + i, tmp + i, (j - i + 1) * sizeof(v[0]));
    }
}

/**
 * Sort v[] of length n using Merge Sort; after allocating a temporary
 * array with the same size of a (used for merging), this function
 * just calls mergesort_rec with the appropriate parameters.  After
 * mergesort_rec terminates, the temporary array is deallocated.
 */
void mergesort(int *v, int n)
{
    int* tmp = (int*) malloc(n * sizeof(v[0]));
    assert(tmp != NULL);
    /* [TODO] Create a parallel region, and make sure that only one
       thread calls mergesort_rec() to start the recursion. */
    #pragma omp parallel default(none) shared(v, n, tmp)
    #pragma omp master
    mergesort_rec(v, 0, n - 1, tmp);
    free(tmp);
}

void print_array(int arr[], int size, FILE *file) {
    assert(file != NULL);
    if (size <= 0) {
        fprintf(file, "[]\n");
        return;
    }

    fprintf(file, "[");
    int count = 0;
    for (int i = 0; i < size - 1; i++) {
        PRINT_RIGHT_ALIGN(file, arr[i]);
        count++;
        if (count != 10) {
            fprintf(file, ", ");
        } else {
            fprintf(file, ",\n ");
            count = 0;
        }
    }
    PRINT_RIGHT_ALIGN(file, arr[size - 1]);
    fprintf(file, "]\n");
}

void create_unsorted_array(int* arr, int size, int lower_bound, int upper_bound) {
    assert(upper_bound >= lower_bound);
#pragma omp parallel default(none) if(size > 100000) shared(arr, size, lower_bound, upper_bound)
    {
        unsigned int seed = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < size; i++) {
            arr[i] = rand_r(&seed) % (upper_bound - lower_bound + 1) + lower_bound;
        }
    }
}

/* Return 1 iff a[] contains the values 0, 1, ... n-1, in that order */
int is_correct(const int* a, int n)
{
    for (int i = 0; i < n - 1; i++) {
        if (a[i] > a[i + 1]) {
            fprintf(stderr, "Expected a[%d] <= a[%d], (it is %d (a[i]) > %d (a[i + 1]))\n", i, i + 1, a[i], a[i + 1]);
            return 0;
        }
    }
    return 1;
}

#define DEFAULT_SIZE 10000000 // int
#define MAX_SIZE 1000000000 // int

int main(int argc, char* argv[])
{
    int n = DEFAULT_SIZE;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (n > MAX_SIZE) {
        fprintf(stderr, "FATAL: array too large\n");
        return EXIT_FAILURE;
    }

    int *a = (int*) malloc(n * sizeof(a[0]));
    assert(a != NULL);

    printf("Initializing array...\n");
    create_unsorted_array(a, n, MIN, MAX);
    if (n <= DEFAULT_SIZE) {
        FILE *file_unsorted = fopen(FILE_UNSORTED, "w");
        if (file_unsorted == NULL) {
            fprintf(stderr, "Error opening file ");
            fprintf(stderr, FILE_UNSORTED);
            return EXIT_FAILURE;
        }
        print_array(a, n, file_unsorted);
        fclose(file_unsorted);
    }
    printf("Sorting %d elements...", n);
    fflush(stdout);
    const double t_start = omp_get_wtime();
    mergesort(a, n);
    const double t_elapsed = omp_get_wtime() - t_start;
    printf("done\n");
    const int ok = is_correct(a, n);
    printf("Check %s\n", (ok ? "OK" : "failed"));
    if (n <= DEFAULT_SIZE && ok) {
        FILE *file_sorted = fopen(FILE_SORTED, "w");
        if (file_sorted == NULL) {
            fprintf(stderr, "Error opening file ");
            fprintf(stderr, FILE_SORTED);
            return EXIT_FAILURE;
        }
        print_array(a, n, file_sorted);
        fclose(file_sorted);
    }
    printf("!! Elapsed time: %.2f s !!\n", t_elapsed);

    free(a);

    return EXIT_SUCCESS;
}
