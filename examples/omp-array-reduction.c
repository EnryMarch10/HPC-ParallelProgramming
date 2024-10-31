/****************************************************************************
 *
 * omp-array-reduction - Demo of reduction on arrays
 *
 * Copyright (C) 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * ----------------------------------------------------------------------------
 *
 * Compile with:
 *
 *      gcc -fopenmp omp-array-reduction.c -o omp-array-reduction
 *
 * Run with:
 *
 *      OMP_NUM_THREADS=1 ./omp-array-reduction
 *      OMP_NUM_THREADS=2 ./omp-array-reduction
 *      OMP_NUM_THREADS=4 ./omp-array-reduction
 *
 ****************************************************************************/

#include <stdio.h>

int main( void )
{
    int m[][4] = {{ 1,  2,  3,  4},
                  { 5,  6,  7,  8},
                  { 9, 10, 11, 12},
                  {13, 14, 15, 15}};
    int row_sum[4] = {0};

#pragma omp parallel for reduction(+:row_sum[:4])
    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            row_sum[i] += m[i][j];
        }
    }
    for (int i=0; i<4; i++) {
        printf("%d ", row_sum[i]);
    }
    printf("\n");
    return 0;
}
