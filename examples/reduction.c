/****************************************************************************
 *
 * reduction.c - Tree-structured reduction
 *
 * Copyright (C) 2019, 2021 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * gcc -std=c99 -Wall -pedantic reduction.c -o reduction
 *
 * Run with:
 * ./reduction
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* Serial implementation of the sum-reduce operator */
float sum_reduce_serial(float *x, size_t n)
{
    float result = 0.0;
    for (int i = 0; i < n; i++) {
        result += x[i];
    }
    return result;
}

/* Tree-structured implementation of the sum-reduce operator; this
   implementation works for any vector length n > 0 */
float sum_reduce(float *x, size_t n)
{
    size_t n2;
    do {
        n2 = (n + 1) / 2; // n/2 arrotondato per eccesso
        for (int i = 0; i < n2; i++) {
            if (i + n2 < n) x[i] += x[i + n2];
        }
        n = n2;
    } while (n2 > 1);
    return x[0];
}

int main(void)
{
    float v[127], result;
    const int n = sizeof(v) / sizeof(v[0]);
    for (int i = 0; i < n; i++) {
        v[i] = i + 1;
    }
    result = sum_reduce(v, n);
    if (fabs(result - n * (n + 1) / 2.0) > 1e-5) {
        fprintf(stderr, "Error: expected %f, got %f\n", n * (n + 1) / 2.0, result);
        return EXIT_FAILURE;
    }
    printf("Test ok\n");
    return EXIT_SUCCESS;
}
