#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <mpi.h>

const int MAX_IT = 100;

/* The __attribute__(( ... )) definition is gcc-specific, and tells
   the compiler that the fields of this structure should not be padded
   or aligned in any way. */
typedef struct __attribute__((__packed__)) {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

/* color gradient from https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia */
const pixel_t colors[] = {
    { 66,  30,  15}, /* r, g, b */
    { 25,   7,  26},
    {  9,   1,  47},
    {  4,   4,  73},
    {  0,   7, 100},
    { 12,  44, 138},
    { 24,  82, 177},
    { 57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201,  95},
    {255, 170,   0},
    {204, 128,   0},
    {153,  87,   0},
    {106,  52,   3} };
const int NCOLORS = sizeof(colors) / sizeof(colors[0]);

/*
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + cx + i*cy;
 *
 * Returns the first `n` such that `z_n > bound`, or `MAX_IT` if `z_n` is below
 * `bound` after `MAX_IT` iterations.
 */
int iterate(float cx, float cy)
{
    float x = 0.0f, y = 0.0f, x_new, y_new;
    int it;
    for (it = 0; it < MAX_IT && x * x + y * y <= 2.0 * 2.0; it++) {
        x_new = x * x - y * y + cx;
        y_new = 2.0 * x * y + cy;
        x = x_new;
        y = y_new;
    }
    return it;
}

/* Draw the rows of the Mandelbrot set from `y_start` (inclusive) to
   `y_end` (excluded) to the bitmap pointed to by `p`. Note that `p`
   must point to the beginning of the bitmap where the portion of
   image will be stored; in other words, this function writes to
   pixels p[0], p[1], ... `x_size` and `y_size` MUST be the sizes
   of the WHOLE image. */
void draw_lines(int y_start, int y_end, pixel_t* p, int x_size, int y_size)
{
    int x, y;
    for (y = y_start; y < y_end; y++) {
        for (x = 0; x < x_size; x++) {
            const float cx = -2.5 + 3.5 * (float) x / (x_size - 1);
            const float cy = 1 - 2.0 * (float) y / (y_size - 1);
            const int v = iterate(cx, cy);
            if (v < MAX_IT) {
                p->r = colors[v % NCOLORS].r;
                p->g = colors[v % NCOLORS].g;
                p->b = colors[v % NCOLORS].b;
            } else {
                p->r = p->g = p->b = 0;
            }
            p++;
        }
    }
}

int main(int argc, char *argv[])
{
    int my_rank, comm_sz;
    FILE *out = NULL;
    const char *fname = "mandelbrot.ppm";
    pixel_t *bitmap = NULL;
    int x_size, y_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc > 1) {
        y_size = atoi(argv[1]);
    } else {
        y_size = 1024;
    }

    x_size = y_size * 1.4;

    /* x_size and y_size are known to all processes */
    int size;
    if (my_rank == 0) {
        out = fopen(fname, "w");
        if (!out) {
            fprintf(stderr, "Error: cannot create %s\n", fname);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Write the header of the output file */
        fprintf(out, "P6\n");
        fprintf(out, "%d %d\n", x_size, y_size);
        fprintf(out, "255\n");

        /* Allocate the complete bitmap */
        size = x_size * y_size;
        bitmap = (pixel_t *) malloc(size * sizeof(*bitmap));
        assert(bitmap != NULL);
    }

    const int my_start = y_size * my_rank / comm_sz;
    const int my_end = y_size * (my_rank + 1) / comm_sz;

    const int local_y_size = my_end - my_start;

    /* Allocate the partial bitmaps */
    const int local_size = x_size * local_y_size;
    pixel_t *local_bitmap = (pixel_t *) malloc(local_size * sizeof(*local_bitmap));
    assert(local_bitmap != NULL);

    draw_lines(my_start, my_end, local_bitmap, x_size, y_size);

    int *recvcounts = NULL;
    int *displs = NULL;
    if (my_rank == 0) {
        recvcounts = (int *) malloc(comm_sz * sizeof(*recvcounts));
        assert(recvcounts != NULL);
        displs = (int *) malloc(comm_sz * sizeof(*displs));
        assert(displs != NULL);

        recvcounts[0] = local_size;
        displs[0] = 0;
        for (int i = 1; i < comm_sz; i++) {
            const int start = y_size * i / comm_sz;
            const int end = y_size * (i + 1) / comm_sz;
            recvcounts[i] = (end - start) * x_size;
            displs[i] = start * x_size;
        }
    }

    int blklen = 3;
    MPI_Aint displ = 0;
    MPI_Datatype oldtype = MPI_UINT8_T, rgb_pixel_t;
    MPI_Type_create_struct(1, &blklen, &displ, &oldtype, &rgb_pixel_t);
    MPI_Type_commit(&rgb_pixel_t);

    MPI_Gatherv(local_bitmap,  /* sendbuf    */
                local_size,    /* sendcount  */
                rgb_pixel_t,   /* sendtype   */
                bitmap,        /* recvbuf    */
                recvcounts,    /* recvcounts */
                displs,        /* displs     */
                rgb_pixel_t,   /* recvtype   */
                0,             /* root       */
                MPI_COMM_WORLD /* comm       */
                );

    MPI_Type_free(&rgb_pixel_t);
    free(local_bitmap);
    free(recvcounts);
    free(displs);

    if (my_rank == 0) {
        fwrite(bitmap, sizeof(*bitmap), size, out);
        fclose(out);
        free(bitmap);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
