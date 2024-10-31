/****************************************************************************
 *
 * opencl-image-rotate.c - Image rotation
 *
 * Copyright (C) 2018, 2021, 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * ---------------------------------------------------------------------------
 *
 * This program rotates a PGM image 90 degrees clockwise using OpenCL.
 * Two kernels are implemented: one that does not use shared memory,
 * and one that does. This program requires that the image is square
 * (i.e., width == height), and that the width/height is a multiple of
 * the local size of the workgroups.
 *
 * Input is read from stdin; output is written to stdout.
 *
 * This program illustrates the importance of collapsing memory
 * accesses to maximize the memory bandwidth. The kernel that uses
 * shared memory might be faster (depending on the underlying
 * hardware) since writes can be collapsed.
 *
 * Compile with:
 *
 * cc opencl-image-rotate.c simpleCL.c -o opencl-image-rotate -lOpenCL
 *
 * Run with:
 *
 * ./opencl-image-rotate 1001 < cat-1344.pgm > out.pgm
 *
 ****************************************************************************/

/* The following #define is required by the implementation of
   hpc_gettime(). It MUST be defined before including any other
   file. */
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "simpleCL.h"
#include "hpc.h"

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Used by the PGM read/write routines */
    unsigned char *bmap; /* buffer of width*height bytes; each byte represents a pixel */
} img_t;

/**
 * Read a PGM file from `f`. This function is not very robust; it may
 * fail on valid PGM images.
 */
void read_pgm( FILE *f, img_t* img )
{
    char buf[2048];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    /* Get the file type (must be "P5") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P5\n")) {
        fprintf(stderr, "FATAL: wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; the code below does not work if
       there are leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxgrey; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxgrey));
    if ( img->maxgrey > 255 ) {
        fprintf(stderr, "FATAL: maxgray > 255 (%d)\n", img->maxgrey);
        exit(EXIT_FAILURE);
    }
    /* Get the binary data */
    img->bmap = (unsigned char*)malloc((img->width)*(img->height));
    nread = fread(img->bmap, 1, (img->width)*(img->height), f);
    if ( (img->width)*(img->height) != nread ) {
        fprintf(stderr, "FATAL: error reading input: expecting %d bytes, got %d\n", (img->width)*(img->height), nread);
        exit(EXIT_FAILURE);
    }
}

/**
 * Write image `img` to `f`
 */
void write_pgm( FILE *f, const img_t* img )
{
    fprintf(f, "P5\n");
    fprintf(f, "# produced by opencl-image-rotate.c\n");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    fwrite(img->bmap, 1, (img->width)*(img->height), f);
}

/**
 * Free the memory allocated by image `img`, and set the additional
 * info to invalid values.
 */
void free_pgm( img_t *img )
{
    img->width = img->height = img->maxgrey = -1;
    free(img->bmap);
    img->bmap = NULL;
}

int main( int argc, char* argv[] )
{
    sclInitFromFile("opencl-image-rotate.cl");

    img_t img;
    cl_mem d_orig, d_new, tmp;
    int nrotations = 1001;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s nrotations < input > output\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        nrotations = atoi(argv[1]);
    }

    read_pgm(stdin, &img);

    if ( img.width != img.height ) {
        fprintf(stderr, "FATAL: width (%d) and height (%d) of the input image must be equal\n", img.width, img.height);
        return EXIT_FAILURE;
    }

    const int n = img.width;

    if (n % SCL_DEFAULT_WG_SIZE2D != 0) {
        fprintf(stderr, "FATAL: width/height (%d) must be a multiple of %d\n", n, (int)SCL_DEFAULT_WG_SIZE2D);
        return EXIT_FAILURE;
    }

    sclKernel rotate_kernel = sclCreateKernel("rotate_kernel");
    sclKernel rotate_kernel_shared = sclCreateKernel("rotate_kernel_shared");

    /* Allocate buffers on the device */
    const size_t size = n*n;
    d_orig = sclMallocCopy(size, img.bmap, CL_MEM_READ_WRITE);
    d_new = sclMalloc(size, CL_MEM_READ_WRITE);

    /* Define block and grid sizes */
    const sclDim block = DIM2(SCL_DEFAULT_WG_SIZE2D, SCL_DEFAULT_WG_SIZE2D);
    const sclDim grid = DIM2(sclRoundUp(img.width, SCL_DEFAULT_WG_SIZE2D),
                             sclRoundUp(img.height, SCL_DEFAULT_WG_SIZE2D));

    fprintf(stderr, "\nPerforming %d rotations (img size %dx%d)\n\n", nrotations, img.width, img.height);

    double tstart = hpc_gettime();
    for (int i=0; i<nrotations; i++) {
        sclSetArgsEnqueueKernel(rotate_kernel,
                                grid, block,
                                ":b :b :d",
                                d_orig, d_new, n);
        tmp = d_orig; d_orig = d_new; d_new = tmp;
    }
    sclDeviceSynchronize();
    const double elapsed_noshmem = hpc_gettime() - tstart;
    const double Mpixels = ((double)img.width * img.height * nrotations)/1.0e6;
    fprintf(stderr, "No shmem: elapsed time %.2f s = %.2f Mpixels/s\n", elapsed_noshmem,
            Mpixels/elapsed_noshmem);

    /* Copy again image to the device, so we are sure that both
       kernels work on the same input (we should not care, since the
       timing is the same). */
    sclMemcpyHostToDevice(d_orig, img.bmap, size);

    tstart = hpc_gettime();
    for (int i=0; i<nrotations; i++) {
        sclSetArgsEnqueueKernel(rotate_kernel_shared,
                                grid, block,
                                ":b :b :d",
                                d_orig, d_new, n);
        tmp = d_orig; d_orig = d_new; d_new = tmp;
    }
    sclDeviceSynchronize();
    const double elapsed_shmem = hpc_gettime() - tstart;
    fprintf(stderr, "Shmem   : elapsed time %.2f s = %.2f Mpixels/s\n", elapsed_shmem,
            Mpixels/elapsed_shmem);

    /* Copy output to host */
    sclMemcpyDeviceToHost(img.bmap, d_orig, size);
    write_pgm(stdout, &img);

    free_pgm(&img);
    sclFree(d_orig);
    sclFree(d_new);
    sclFinalize();
    return EXIT_SUCCESS;
}
