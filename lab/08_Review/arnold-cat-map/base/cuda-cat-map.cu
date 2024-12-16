/****************************************************************************
 *
 * cuda-cat-map.cu - Arnold's cat map
 *
 * Copyright (C) 2016--2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "hpc.h"

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Don't care (used only by the PGM read/write routines) */
    unsigned char *bmap; /* buffer of width*height bytes; each element represents the gray level of a pixel (0-255) */
} PGM_image;

const unsigned char WHITE = 255;
const unsigned char BLACK = 0;

/**
 * Initialize a PGM_image object: allocate space for a bitmap of size
 * `width` x `height`, and set all pixels to color `col`
 */
void init_pgm(PGM_image *img, int width, int height, unsigned char col)
{
    int i, j;

    assert(img != NULL);

    img->width = width;
    img->height = height;
    img->maxgrey = 255;
    img->bmap = (unsigned char *) malloc(width * height);
    assert(img->bmap != NULL);
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            img->bmap[i * width + j] = col;
        }
    }
}

/**
 * Read a PGM file from file `f`. Warning: this function is not
 * robust: it may fail on legal PGM images, and may crash on invalid
 * files since no proper error checking is done.
 */
void read_pgm(FILE *f, PGM_image *img)
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P5") */
    s = fgets(buf, BUFSIZE, f);
    if (strcmp(s, "P5\n") != 0) {
        fprintf(stderr, "Wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxgrey; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxgrey));
    if (img->maxgrey > 255) {
        fprintf(stderr, "FATAL: maxgray=%d > 255\n", img->maxgrey);
        exit(EXIT_FAILURE);
    }
#if _XOPEN_SOURCE < 600
    img->bmap = (unsigned char *) malloc(img->width * img->height * sizeof(unsigned char));
#else
    /* The pointer img->bmap must be properly aligned to allow aligned
       SIMD load/stores to work. */
    int ret = posix_memalign((void**) &(img->bmap), __BIGGEST_ALIGNMENT__, img->width * img->height);
    assert(ret == 0);
#endif
    assert(img->bmap != NULL);
    /* Get the binary data from the file */
    nread = fread(img->bmap, 1, img->width * img->height, f);
    if (img->width * img->height != nread) {
        fprintf(stderr, "FATAL: error reading input: expecting %d bytes, got %d\n", img->width * img->height, nread);
        exit(EXIT_FAILURE);
    }
}

/**
 * Write the image `img` to file `f`; if not NULL, use the string
 * `comment` as metadata.
 */
void write_pgm(FILE *f, const PGM_image* img, const char *comment)
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P5\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    fwrite(img->bmap, 1, img->width * img->height, f);
}

/**
 * Free the bitmap associated with image `img`; note that the
 * structure pointed to by `img` is NOT deallocated; only `img->bmap`
 * is.
 */
void free_pgm(PGM_image *img)
{
    assert(img != NULL);
    free(img->bmap);
    img->bmap = NULL; /* not necessary */
    img->width = img->height = img->maxgrey = -1;
}


/**
 * Compute the `k`-th iterate of the cat map for image `img`. The
 * width and height of the input image must be equal. This function
 * replaces the bitmap of `img` with the one resulting after iterating
 * `k` times the cat map. You need to allocate a temporary image, with
 * the same size of the original one, so that you read the pixel from
 * the "old" image and copy them to the "new" image (this is similar
 * to a stencil computation, as was discussed in class). After
 * applying the cat map to all pixel of the "old" image the role of
 * the two images is exchanged: the "new" image becomes the "old" one,
 * and vice-versa. The temporary image must be deallocated upon exit.
 */
void cat_map(PGM_image *img, int k)
{
    const int N = img->width;
    const size_t size = N * N * sizeof(img->bmap[0]);

    /* [TODO] Modify the body of this function to allocate device memory,
       do the appropriate data transfer, and launch a kernel */
    unsigned char *cur = img->bmap;
    unsigned char *next = (unsigned char *) malloc(size);

    assert(next != NULL);
    for (int i = 0; i < k; i++) {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                int xnext = (2 * x + y) % N;
                int ynext = (x + y) % N;
                next[xnext + ynext * N] = cur[x + y * N];
            }
        }
        /* Swap old and new */
        unsigned char *tmp = cur;
        cur = next;
        next = tmp;
    }
    img->bmap = cur;
    free(next);
}

int main(int argc, char *argv[])
{
    PGM_image img;
    int niter;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s niter < input_image > output_image\n", argv[0]);
        return EXIT_FAILURE;
    }
    niter = atoi(argv[1]);
    read_pgm(stdin, &img);
    if (img.width != img.height) {
        fprintf(stderr, "FATAL: width (%d) and height (%d) of the input image must be equal\n", img.width, img.height);
        return EXIT_FAILURE;
    }
    const double tstart = hpc_gettime();
    cat_map(&img, niter);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "      Iterations : %d\n", niter);
    fprintf(stderr, "    width,height : %d,%d\n", img.width, img.height);
    fprintf(stderr, "        Mops/sec : %.4f\n", 1.0e-6 * img.width * img.height * niter / elapsed);
    fprintf(stderr, "Elapsed time (s) : %f\n", elapsed);

    write_pgm(stdout, &img, "produced by cuda-cat-map.cu");
    free_pgm(&img);
    return EXIT_SUCCESS;
}
