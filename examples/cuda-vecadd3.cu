/****************************************************************************
 *
 * cuda-vecadd3.cu - Sum two arrays with CUDA, using threads and blocks
 *
 * Based on the examples from the CUDA toolkit documentation
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/
 *
 * Last updated in 2017 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 *
 *      nvcc cuda-vecadd3.cu -o cuda-vecadd3
 *
 * Run with:
 *
 *      ./cuda-vecadd3
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *a, int *b, int *c, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

void vec_init(int *a, int n)
{
    for (int i = 0; i < n; i++) {
        a[i] = i;
    }
}

#define N (2048 * 2048)
#define BLKDIM 1024

int main(void)
{
    int *a, *b, *c;	          /* host copies of a, b, c */
    int *d_a, *d_b, *d_c;     /* device copies of a, b, c */
    int i;
    const size_t size = N * sizeof(int);

    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    /* Allocate space for host copies of a, b, c */
    a = (int *) malloc(size);
    vec_init(a, N);
    b = (int *) malloc(size);
    vec_init(b, N);
    c = (int *) malloc(size);

    /* Copy inputs to device */
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    /* Launch add() kernel on GPU */
    printf("Adding %d elements\n", N);
    add<<<(N + BLKDIM - 1) / BLKDIM, BLKDIM>>>(d_a, d_b, d_c, N);

    /* Copy result back to host */
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    /* Check results */
    for (i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            fprintf(stderr, "Error at index %d: a[%d]=%d, b[%d]=%d, c[%d]=%d\n",
                    i, i, a[i], i, b[i], i, c[i]);
            break;
        }
    }
    if (i == N) {
        printf("Check OK\n");
    }
    /* Cleanup */
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return EXIT_SUCCESS;
}
