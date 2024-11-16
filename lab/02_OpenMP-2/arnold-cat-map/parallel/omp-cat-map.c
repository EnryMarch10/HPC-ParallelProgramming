#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include "pgm.h"

/**
 * Compute the `k`-th iterate of the cat map for image `img`. The
 * width and height of the image must be equal. This function must
 * replace the bitmap of `img` with the one resulting after iterating
 * `k` times the cat map. To do so, the function allocates a temporary
 * bitmap with the same size of the original one, so that it reads one
 * pixel from the "old" image and copies it to the "new" image. After
 * each iteration of the cat map, the role of the two bitmaps are
 * exchanged.
 */
void cat_map(PGM_image *img, int k)
{
    const int N = img->width;
    unsigned char *cur = img->bmap;
    unsigned char *next = (unsigned char *) malloc(N * N * sizeof(unsigned char));
    unsigned char *tmp;

    assert(next != NULL);
    assert(img->width == img->height);

    /* [TODO] Which of the following loop(s) can be parallelized? */
    for (int i = 0; i < k; i++) {
#pragma omp parallel for collapse(2) default(none) shared(N, next, cur)
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                const int x_next = (2 * x + y) % N;
                const int y_next = (x + y) % N;
                next[y_next * N + x_next] = cur[x + y * N];
            }
        }
        /* Swap old and new */
        tmp = cur;
        cur = next;
        next = tmp;
    }
    img->bmap = cur;
    free(next);
}

void cat_map_interchange(PGM_image *img, int k)
{
    const int N = img->width;
    unsigned char *cur = img->bmap;
    unsigned char *next = (unsigned char *) malloc(N * N * sizeof(unsigned char));

    assert(next != NULL);
    assert(img->width == img->height);

    /* [TODO] Which of the following loop(s) can be parallelized? */
#pragma omp parallel for collapse(2) default(none) shared(N, next, cur, k)
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            /* Compute the k-th iterate of pixel (x, y) */
            int x_cur = x, y_cur = y;
            for (int i = 0; i < k; i++) {
                const int x_next = (2 * x_cur + y_cur) % N;
                const int y_next = (x_cur + y_cur) % N;
                x_cur = x_next;
                y_cur = y_next;
            }
            next[y_cur * N + x_cur] = cur[y * N + x];
        }
    }
    img->bmap = next;
    free(cur);
}

int main(int argc, char *argv[])
{
    PGM_image img;
    int niter;
    double t_elapsed;
    const int N_TESTS = 5; /* number of replications */

    if (argc != 2) {
        fprintf(stderr, "Usage: %s niter < input > output\n", argv[0]);
        return EXIT_FAILURE;
    }
    niter = atoi(argv[1]);
    read_pgm(stdin, &img);

    if (img.width != img.height) {
        fprintf(stderr, "FATAL: width (%d) and height (%d) of the input image must be equal\n", img.width, img.height);
        return EXIT_FAILURE;
    }

    /**
     ** WITHOUT loop interchange
     **/
    t_elapsed = 0.0;
    for (int i = 0; i < N_TESTS; i++) {
        fprintf(stderr, "Run %d of %d\n", i + 1, N_TESTS);
        const double t_start = hpc_gettime();
        cat_map(&img, niter);
        t_elapsed += hpc_gettime() - t_start;
        if (i == 0) {
            write_pgm(stdout, &img, "produced by omp-cat-map.c");
        }
    }
    t_elapsed /= N_TESTS;

    fprintf(stderr, "\n=== Without loop interchange ===\n");
#if defined(_OPENMP)
    fprintf(stderr, "  OpenMP threads: %d\n", omp_get_max_threads());
#else
    fprintf(stderr, "  OpenMP disabled\n");
#endif
    fprintf(stderr, "     Iterations: %d\n", niter);
    fprintf(stderr, "   width,height: %d,%d\n", img.width, img.height);
    fprintf(stderr, "       Mops/sec: %f\n", 1.0e-6 * img.width * img.height * niter / t_elapsed);
    fprintf(stderr, "!! Elapsed time: %.2f s !!\n\n", t_elapsed);

    /**
     ** WITH loop interchange
     **/
    t_elapsed = 0.0;
    for (int i = 0; i < N_TESTS; i++) {
        fprintf(stderr, "Run %d of %d\n", i + 1, N_TESTS);
        const double t_start = hpc_gettime();
        cat_map_interchange(&img, niter);
        t_elapsed += hpc_gettime() - t_start;
    }
    t_elapsed /= N_TESTS;

    fprintf(stderr, "\n=== With loop interchange ===\n");
#if defined(_OPENMP)
    fprintf(stderr, "  OpenMP threads: %d\n", omp_get_max_threads());
#else
    fprintf(stderr, "  OpenMP disabled\n");
#endif
    fprintf(stderr, "     Iterations: %d\n", niter);
    fprintf(stderr, "   width,height: %d,%d\n", img.width, img.height);
    fprintf(stderr, "       Mops/sec: %.4f\n", 1.0e-6 * img.width * img.height * niter / t_elapsed);
    fprintf(stderr, "!! Elapsed time: %.2f s !!\n\n", t_elapsed);

    free_pgm(&img);
    return EXIT_SUCCESS;
}
