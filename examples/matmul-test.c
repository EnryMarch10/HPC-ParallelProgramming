/****************************************************************************
 *
 * matmul-test.c - Dense matrix-matrix multiply
 *
 * Copyright (C) 2017, 2022, 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 *
 *      gcc -O2 -mavx2 -mfma -fopenmp matmul-test.c -o matmul-test
 *
 * Run with:
 *
 *      ./matmul-test [n]
 *
 * (n = matrix size).
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>
#include <immintrin.h>

typedef void (*matmul_algo_t)(const float *, const float *, float *, int);
#define VLEN sizeof(__m256) / sizeof(float)

/* Fills n x n square matrix m with random values */
void fill(float *m, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i * n + j] = rand() / (double) RAND_MAX; /* cast to float */
        }
    }
}

#define MATMUL_SEQ_BODY(I1, I2, I3) \
{\
    for (int i = 0; i < n; i++) {\
        for (int j = 0; j < n; j++) {\
            r[i * n + j] = 0;\
        }\
    }\
\
    for (int I1 = 0; I1 < n; I1++) {\
        for (int I2 = 0; I2 < n; I2++) {\
            for (int I3 = 0; I3 < n; I3++) {\
                r[i * n + j] += p[i * n + k] * q[k * n + j];\
            }\
        }\
    }\
}

void matmul_seq_ijk(const float *p, const float *q, float *r, int n) { MATMUL_SEQ_BODY(i, j, k); }
void matmul_seq_ijk_restrict(const float * restrict p, const float * restrict q, float * restrict r, int n) { MATMUL_SEQ_BODY(i, j, k); }
void matmul_seq_ikj(const float * restrict p, const float * restrict q, float * restrict r, int n) { MATMUL_SEQ_BODY(i, k, j); }
void matmul_seq_jik(const float * restrict p, const float * restrict q, float * restrict r, int n) { MATMUL_SEQ_BODY(j, i, k); }
void matmul_seq_jki(const float * restrict p, const float * restrict q, float * restrict r, int n) { MATMUL_SEQ_BODY(j, k, i); }
void matmul_seq_kij(const float * restrict p, const float * restrict q, float * restrict r, int n) { MATMUL_SEQ_BODY(k, i, j); }
void matmul_seq_kji(const float * restrict p, const float * restrict q, float * restrict r, int n) { MATMUL_SEQ_BODY(k, j, i); }

/* Cache-efficient computation of r = p * q, where p. q, r are n x n
   matrices. The caller is responsible for allocating the memory for
   r. This function allocates (and the frees) an additional n x n
   temporary matrix. */
void matmul_seq_transpose(const float * restrict p, const float * restrict q, float * restrict r, int n)
{
    float * restrict qT = (float *) malloc(n * n * sizeof(float));

    /* transpose q, storing the result in qT */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            qT[j * n + i] = q[i * n + j];
        }
    }

    /* multiply p and qT row-wise */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int k = 0; k < n; k++) {
                s += p[i * n + k] * qT[j * n + k];
            }
            r[i * n + j] = s;
        }
    }

    free(qT);
}

void matmul_omp_ijk(const float * restrict p, const float * restrict q, float * restrict r, int n)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0.0;
            for (int k = 0; k < n; k++) {
                s += p[i * n + k] * q[k * n + j];
            }
            r[i * n + j] = s;
        }
    }
}

void matmul_omp_ikj(const float * restrict p, const float * restrict q, float * restrict r, int n)
{
#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                r[i * n + j] = 0;
            }
        }

#pragma omp for
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++) {
                    r[i * n + j] += p[i * n + k] * q[k * n + j];
                }
            }
        }
    }
}

void matmul_simd_ikj(const float *p, const float *q, float *r, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j += VLEN) {
            __m256 *ptr = (__m256 *) &r[i * n + j];
            *ptr = _mm256_setzero_ps();;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            const __m256 pv = _mm256_broadcast_ss(&p[i * n + k]);
            for (int j = 0; j < n; j += VLEN) {
                __m256 rv = _mm256_load_ps(&r[i * n + j]);
                const __m256 qv = _mm256_load_ps(&q[k * n + j]);
                rv = _mm256_fmadd_ps(pv, qv, rv);
                _mm256_store_ps(&r[i * n + j], rv);
                // r[i * n + j] += p[i * n + k] * q[k * n + j];
            }
        }
    }
}

void matmul_omp_simd_ikj(const float *p, const float *q, float *r, int n)
{
#pragma omp parallel default(none) shared(n, p, q, r)
    {
#pragma omp for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j += VLEN) {
                __m256 *ptr = (__m256 *) &r[i * n + j];
                *ptr = _mm256_setzero_ps();;
            }
        }

#pragma omp for
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                const __m256 pv = _mm256_broadcast_ss(&p[i * n + k]);
                for (int j = 0; j < n; j += VLEN) {
                    __m256 rv = _mm256_load_ps(&r[i * n + j]);
                    const __m256 qv = _mm256_load_ps(&q[k * n + j]);
                    rv = _mm256_fmadd_ps(pv, qv, rv);
                    _mm256_store_ps(&r[i * n + j], rv);
                    // r[i * n + j] += p[i * n + k] * q[k * n + j];
                }
            }
        }
    }
}

void matmul_omp_transpose(const float * restrict p, const float * restrict q, float * restrict r, int n)
{
    float * restrict qT = (float *) malloc(n * n * sizeof(float));

    /* transpose q, storing the result in qT */
#pragma omp parallel default(none) shared(n, p, q, r, qT)
    {
#pragma omp for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                qT[j * n + i] = q[i * n + j];
            }
        }

        /* multiply p and qT row-wise */
#pragma omp for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += p[i * n + k] * qT[j * n + k];
                }
                r[i * n + j] = s;
            }
        }
    }

    free(qT);
}

void matmul_omp_simd_transpose(const float *p, const float *q, float *r, int n)
{
    /* Ensure that the matrix size is multiple of VLEN */
    float * restrict qT;
    int err;

    assert(n % VLEN == 0);

    err = posix_memalign((void**) &qT, __BIGGEST_ALIGNMENT__, n * n * sizeof(float));
    assert(err == 0);

    /* transpose q, storing the result in qT */
#pragma omp parallel default(none) shared(n, p, q, r, qT)
    {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                qT[j * n + i] = q[i * n + j];
            }
        }

        /* multiply p and qT row-wise */
#pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                __m256 vvs = _mm256_setzero_ps();
                for (int k = 0; k < n; k += VLEN) {
                    __m256 vvp = _mm256_load_ps(&p[i * n + k]);
                    __m256 vvqT = _mm256_load_ps(&qT[j * n + k]);
                    vvs = _mm256_fmadd_ps(vvp, vvqT, vvs);
                }
                r[i * n + j] = vvs[0] + vvs[1] + vvs[2] + vvs[3] + vvs[4] + vvs[5] + vvs[6] + vvs[7];
            }
        }
    }

    free(qT);
}

int main(int argc, char *argv[])
{
    int err;
    int n = 4096;
    float *p, *q, *r;
    struct {
        matmul_algo_t algo;
        char *description;
    } matmul_algos[] = { {matmul_seq_ijk, "Sequential ijk"},
                         {matmul_seq_ijk_restrict, "Sequential ijk restrict"},
                         {matmul_seq_ikj, "Sequential ikj"},
                         {matmul_seq_jik, "Sequential jik"},
                         {matmul_seq_jki, "Sequential jki"},
                         {matmul_seq_kij, "Sequential kij"},
                         {matmul_seq_kji, "Sequential kji"},
                         {matmul_simd_ikj, "SIMD ikj"},
                         //{matmul_seq_transpose, "Sequential transposed"},
                         {matmul_omp_ijk, "OpenMP ijk"},
                         {matmul_omp_ikj, "OpenMP ikj"},
                         {matmul_omp_simd_ikj, "OpenMP+SIMD ikj"},
                         //{matmul_omp_transpose, "OpenMP transposed"},
                         //{matmul_omp_simd_transpose, "OpenMP+SIMD transposed"},
                         {NULL, NULL} };
    if (argc > 2) {
        printf("Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 2) {
        n = atoi(argv[1]);
    }

    const size_t MAT_SIZE = n * n * sizeof(*p);

    err = posix_memalign((void **) &p, __BIGGEST_ALIGNMENT__, MAT_SIZE);
    err += posix_memalign((void **) &q, __BIGGEST_ALIGNMENT__, MAT_SIZE);
    err += posix_memalign((void **) &r, __BIGGEST_ALIGNMENT__, MAT_SIZE);
    assert(err == 0);

    fill(p, n);
    fill(q, n);

    printf("Matrix-Matrix multiplication (C) %d x %d\n\n", n, n);

    printf("Algorithm                     \t  Time (s)\t    Gflops\n");
    printf("------------------------------\t----------\t----------\n");
    for (int a = 0; matmul_algos[a].algo != NULL; a++) {
        printf("%-30s\t", matmul_algos[a].description); fflush(stdout);
        /* Purge matrix data from the cache;
           `__builtin___clear_cache()` is a built-in function of the
           GCC compiler that acts as a portable interface for the
           appropriate low-level OS function, if available. */
        __builtin___clear_cache(p, p + MAT_SIZE);
        __builtin___clear_cache(q, q + MAT_SIZE);
        __builtin___clear_cache(r, r + MAT_SIZE);
        const double tstart = omp_get_wtime();
        matmul_algos[a].algo(p, q, r, n);
        const double elapsed = omp_get_wtime() - tstart;
        const double Gflops = 2.0 * (n / 1000.0) * (n / 1000.0) * (n / 1000.0) / elapsed;
        printf("%10.3f\t%10.3f\n", elapsed, Gflops);
    }


    free(p);
    free(q);
    free(r);

    return EXIT_SUCCESS;
}
