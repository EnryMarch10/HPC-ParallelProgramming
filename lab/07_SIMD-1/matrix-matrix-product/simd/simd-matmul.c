/* The following #define is required by posix_memalign() */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>  /* for assert() */
#include <strings.h> /* for bzero() */

#include "hpc.h"

/* This program works on double-precision numbers; therefore, we
   define a v2d vector datatype that contains two doubles in a SIMD
   array of 16 bytes (VLEN==2). */
typedef double vxd __attribute__((vector_size(16)));
#define VLEN (sizeof(vxd) / sizeof(double))

/* Fills n x n square matrix m */
void fill(double *m, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i * n + j] = (i % 10 + j) / 10.0;
        }
    }
}

/* compute r = p * q, where p, q, r are n x n matrices. */
void scalar_matmul(const double *p, const double *q, double *r, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += p[i * n + k] * q[k * n + j];
            }
            r[i * n + j] = s;
        }
    }
}

/* Cache-efficient computation of r = p * q, where p. q, r are n x n
   matrices. This function allocates (and then releases) an additional n x n
   temporary matrix. */
void scalar_matmul_tr(const double *p, const double *q, double *r, int n)
{
    double *qT = (double *) malloc(n * n * sizeof(*qT));

    assert(qT != NULL);

    /* transpose q, storing the result in qT */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            qT[j * n + i] = q[i * n + j];
        }
    }

    /* multiply p and qT row-wise */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += p[i * n + k] * qT[j * n + k];
            }
            r[i * n + j] = s;
        }
    }

    free(qT);
}

/* SIMD version of the cache-efficient matrix-matrix multiply above.
   This function requires that n is a multiple of the SIMD vector
   length VLEN */
void simd_matmul_tr(const double *p, const double *q, double *r, int n)
{
    assert(n % VLEN == 0);

    double *qT;
    assert(posix_memalign((void **) &qT, __BIGGEST_ALIGNMENT__, n * n * sizeof(*qT)) == 0);

    /* transpose q, storing the result in qT */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            qT[j * n + i] = q[i * n + j];
        }
    }

    /* multiply p and qT row-wise */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vxd vs = {0.0};
            const vxd *vp_r = (vxd *) (p + i * n);
            const vxd *vqT_r = (vxd *) (qT + i * n);
            for (int k = 0; k < n - VLEN + 1; k += VLEN) {
                vs += *vp_r * *vqT_r;
                vp_r++;
                vqT_r++;
            }
            r[i * n + j] = 0.0;
            for (int h = 0; h < VLEN; h++) {
                r[i * n + j] += vs[h];
            }
        }
    }

    free(qT);
}

int main(int argc, char *argv[])
{
    int n = 512;
    double *p, *q, *r;
    double tstart, elapsed, tserial;
    int ret;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (n % VLEN != 0) {
        fprintf(stderr, "FATAL: the matrix size must be a multiple of %d\n", (int) VLEN);
        return EXIT_FAILURE;
    }

    const size_t size = n * n * sizeof(*p);

    ret = posix_memalign((void **) &p, __BIGGEST_ALIGNMENT__, size);
    assert(0 == ret);
    ret = posix_memalign((void **) &q, __BIGGEST_ALIGNMENT__, size);
    assert(0 == ret);
    ret = posix_memalign((void **) &r, __BIGGEST_ALIGNMENT__, size);
    assert(0 == ret);

    fill(p, n);
    fill(q, n);
    printf("\nMatrix size: %d x %d\n\n", n, n);

    tstart = hpc_gettime();
    scalar_matmul(p, q, r, n);
    tserial = elapsed = hpc_gettime() - tstart;
    printf("Scalar\t\tr[0][0] = %f, Exec time = %f\n", r[0], elapsed);

    bzero(r, size);

    tstart = hpc_gettime();
    scalar_matmul_tr(p, q, r, n);
    elapsed = hpc_gettime() - tstart;
    printf("Transposed\tr[0][0] = %f, Exec time = %f (speedup vs scalar %.2fx)\n", r[0], elapsed, tserial / elapsed);

    bzero(r, size);

    tstart = hpc_gettime();
    simd_matmul_tr(p, q, r, n);
    elapsed = hpc_gettime() - tstart;
    printf("SIMD transposed\tr[0][0] = %f, Exec time = %f (speedup vs scalar %.2fx)\n", r[0], elapsed, tserial / elapsed);

    free(p);
    free(q);
    free(r);
    return EXIT_SUCCESS;
}
