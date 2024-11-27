# Array reversal

Write a program that reverses an array `v[]` of length $n$, i.e., exchanges `v[0]` and `v[n-1]`, `v[1]` and `v[n-2]` and so on.
You should write the following functions:

1. `reverse()` reverses an array `in[]` into a different array `out[]` (the input is not modified). Assume that `in[]` and `out[]`
   reside on non-overlapping memory blocks.
2. `inplace_reverse()` reverses the array `in[]` "in place", i.e., exchanging elements using $O(1)$ additional storage; therefore,
   you are not allowed to allocate a temporary output vector.

The file [cuda-reverse.cu](base/cuda-reverse.cu) provides a serial implementation of `reverse()` and `inplace_reverse()`.
Your goal is to modify the functions to use of the GPU, defining any additional kernel that is required.

## Hints

`reverse()` can be parallelized by launching $n$ CUDA threads; each thread copies a single element from the input to the output
array. Since the array size $n$ can be large, you should create as many one-dimensional _thread blocks_ as needed to have at least
$n$ threads. Have a look at the lecture notes on how to do this.

`inplace_reverse()` can be parallelized by launching $\lfloor n/2 \rfloor$ CUDA threads (note the rounding): each thread swaps an
element on the first half of `in[]` with the corresponding element on the second half.

To map threads to array elements it is possible to use the expression:

```C
const int idx = threadIdx.x + blockIdx.x * blockDim.x;
```

In both cases the program might create more threads than actually needed; special care should be made to ensure that the extra
threads do nothing, e.g., using:

```C
if (idx < n) {
    /* body */
}
/* else do nothing */
```

for `reverse()`, and:

```C
if (idx < n / 2) {
    /* body */
}
/* else do nothing */
```

for `inplace_reverse()`.

To compile:

```shell
nvcc cuda-reverse.cu -o cuda-reverse
```

To execute:

```shell
./cuda-reverse [n]
```

Example:

```shell
./cuda-reverse
```

## Files

- [cuda-reverse.cu](base/cuda-reverse.cu)
- [hpc.h](../../include/hpc.h)