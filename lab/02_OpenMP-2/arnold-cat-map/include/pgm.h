#ifndef __PGM__
#define __PGM__

#include <stdio.h>

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Don't care (used only by the PGM read/write routines) */
    unsigned char *bmap; /* buffer of width * height bytes; each element represents the gray level of a pixel (0-255) */
} PGM_image;

void read_pgm(FILE *f, PGM_image *img);
void write_pgm(FILE *f, const PGM_image *img, const char *comment);
void free_pgm(PGM_image *img);

#endif
