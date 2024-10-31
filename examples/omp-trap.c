/****************************************************************************
 *
 * omp-trap0.c - Trapezoid rule with OpenMP
 *
 * Last updated in 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * The original copyright notice follows.
 *
 * --------------------------------------------------------------------------
 *
 * Copyright (c) 2000, 2013, Peter Pacheco and the University of San
 * Francisco. All rights reserved. Redistribution and use in source
 * and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the
 * distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 *
 *      gcc -std=c99 -Wall -Wpedantic -fopenmp omp-trap.c -o omp-trap
 *
 * Run with:
 *
 *      OMP_NUM_THREADS=4 ./omp-trap [a b n]
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
 * Function to be integrated
 */
double f( double x )
{
    return 4.0/(1.0 + x*x);
}

/*
 * Compute the area of function f(x) for x=[a, b] using the trapezoid
 * rule. The integration interval [a,b] is partitioned into n
 * subintervals of equal size.
 */
double trap( double a, double b, long n )
{
    /*
     * This code is a direct implementation of the trapezoid rule.
     * The area of the trapezoid on interval [x, x+h] is computed as
     * h*(f(x) + f(x+h))/2.0. All areas are accumulated in
     * variable |result|.
     */
    double result = 0.0;
    const double h = (b-a)/n;
    double x = a;
    for ( long i = 0; i<n; i++ ) {
	result += h*(f(x) + f(x+h))/2.0;
	x += h;
    }
    return result;
}

/* Version 0: no pragma omp for, manual reduction */
double trap0( double a, double b, long n )
{
    const int thread_count = omp_get_max_threads(); /* <= Note: omp_get_MAX_threads() */
    double partial_result[thread_count];
#pragma omp parallel default(none) shared(a, b, n, partial_result)
    {
        const int my_rank = omp_get_thread_num();
        const int thread_count = omp_get_num_threads();
        const double h = (b-a)/n;
        const long local_n_start = ((long)n * my_rank) / thread_count;
        const long local_n_end = ((long)n * (my_rank+1)) / thread_count;
        double x = a + local_n_start * h;
        double local_result = 0.0;

        for ( long i = local_n_start; i<local_n_end; i++ ) {
            local_result += h*(f(x) + f(x+h))/2.0;
            x += h;
        }
        partial_result[my_rank] = local_result;
    }
    /* Implicit barrier here */
    /* Only one thread (the master) computes the sum of partial results */
    double result = 0.0;
    for (int i=0; i<thread_count; i++) {
        result += partial_result[i];
    }
    return result;
}

/* Version 1: no pragma omp for, reduction using atomic construct. */
double trap1( double a, double b, long n )
{
    double result = 0.0;
#pragma omp parallel default(none) shared(result, a, b, n)
    {
        const int my_rank = omp_get_thread_num();
        const int thread_count = omp_get_num_threads();
        const double h = (b-a)/n;
        const long local_n_start = ((long)n * my_rank) / thread_count;
        const long local_n_end = ((long)n * (my_rank+1)) / thread_count;
        double x = a + local_n_start * h;
        double partial_result = 0.0;

        for ( long i = local_n_start; i<local_n_end; i++ ) {
            partial_result += h*(f(x) + f(x+h))/2.0;
            x += h;
        }
#pragma omp atomic
        result += partial_result;
    }
    /* Implicit barrier here */
    return result;
}

/* Version 2: no pragma omp for, "proper" reduction. */
double trap2( double a, double b, long n )
{
    double result = 0.0;
#pragma omp parallel default(none) shared(a, b, n) reduction(+:result)
    {
        const int my_rank = omp_get_thread_num();
        const int thread_count = omp_get_num_threads();
        const double h = (b-a)/n;
        const long local_n_start = ((long)n * my_rank) / thread_count;
        const long local_n_end = ((long)n * (my_rank+1)) / thread_count;
        double x = a + local_n_start * h;

        for ( long i = local_n_start; i<local_n_end; i++ ) {
            result += h*(f(x) + f(x+h))/2.0;
            x += h;
        }
    }
    /* Implicit barrier here */
    return result;
}

/* Version 3: pragma omp for with reduction. */
double trap3( double a, double b, long n )
{
    double result = 0.0;
    const double h = (b-a)/n;
#pragma omp parallel for default(none) shared(a, b, n, h) reduction(+:result)
    for ( long i = 0; i<n; i++ ) {
	result += h*(f(a+i*h) + f(a+(i+1)*h))/2;
    }
    return result;
}

typedef double (* trap_fun_t)(double, double, long);

void test(trap_fun_t f, const char *desc, double a, double b, long n)
{
    printf("%s\n", desc);
    const double tstart = omp_get_wtime();
    const double result = f(a, b, n);
    const double elapsed = omp_get_wtime() - tstart;
    printf("\tArea: %f\n\tElapsed time (s): %f\n\n", result, elapsed);
}

int main( int argc, char* argv[] )
{
    long n = 1000000;
    double a = 0.0, b = 1.0;

    if ( 4 == argc ) {
	a = atof(argv[1]);
	b = atof(argv[2]);
	n = atol(argv[3]);
    }

    test(trap, "Sequential", a, b, n);
    test(trap0, "trap0 (parallel, no for, no reduction)", a, b, n);
    test(trap1, "trap1 (parallel, no for, atomic)", a, b, n);
    test(trap2, "trap2 (parallel, no for, reduction)", a, b, n);
    test(trap3, "trap3 (parallel, for, reduction)", a, b, n);

    return EXIT_SUCCESS;
}
