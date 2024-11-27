# Odd-even transposition sort

The _Odd-Even sort_ algorithm is a variant of BubbleSort, and sorts an array of $n$ elements in sequential time $O(n^2)$.
Although inefficient, odd-even sort is easily parallelizable; indeed, we have discussed both an OpenMP and an MPI version.
In this exercise we will create a CUDA version.

Given an array `v[]` of length $n$, the algorithm performs $n$ steps numbered $0, \ldots, n - 1$.
During even steps, array elements in even positions are compared with the next element and swapped if not in the correct order.
During odd steps, elements in odd position are compared (and possibly swapped) with their successors.
See *Figure 1*.

![Figure 1: Odd-Even Sort](img/cuda-odd-even.svg)

*Figure 1: Odd-Even Sort*

The file [cuda-odd-even.cu](base/cuda-odd-even.cu) contains a serial implementation of Odd-Even transposition sort.
The purpose of this algorithm is to modify the program to use the GPU.

The CUDA paradigm suggests a fine-grained parallelism where a CUDA thread is responsible for a single compare-and-swap operation
of a pair of adjacent elements.
The simplest solution is to launch $n$ CUDA threads during each phase; only even (resp. odd) threads will be active during even
(resp. odd) phases.
The kernel looks like this:

```C
__global__ void odd_even_step_bad(int *x, int n, int phase)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n - 1 && idx % 2 == phase % 2) {
		cmp_and_swap(&x[idx], &x[idx + 1]);
	}
}
```

This solution is simple but definitely _not_ efficient since only half the threads are active during each phase, so a lot of
computational resources are wasted.
To address this issue, write a second version where $\lceil n / 2 \rceil$ CUDA threads are executed at each phase, so that each
one is always active.
Indexing becomes more problematic, since each thread should be uniquely assigned to an even (resp. odd) position depending on the
phase.
Specifically, in even phases threads $0, 1, 2, 3, \ldots$ are required to handle the pairs $(0, 1)$, $(2, 3)$, $(4, 5)$, $(6, 7)$,
$\ldots$.
During odd phases, the threads are required to handle the pairs $(1, 2)$, $(3, 4)$, $(5, 6)$, $(7, 8)$, $\ldots$.

*Table 1* illustrates the correspondence between the "linear" index `idx` of each thread, computed using the expression in the
above code snipped, and the index pair it needs to manage.

*Table 1 Mapping thread index to array index pairs.*

| Thread index      | Even phases  | Odd phases     |
| :---------------: | :----------: | :------------: |
| $0$               | $(0,1)$      | $(1,2)$        |
| $1$               | $(2,3)$      | $(3,4)$        |
| $2$               | $(4,5)$      | $(5,6)$        |
| $3$               | $(6,7)$      | $(7,8)$        |
| $4$               | $(8,9)$      | $(9,10)$       | 
| $\ldots$          | $\ldots$     | $\ldots$       |

To compile:

```shell
nvcc cuda-odd-even.cu -o cuda-odd-even
```

To execute:

```shell
./cuda-odd-even [len]
```

Example:

```shell
./cuda-odd-even 1024
```

## Files

- [cuda-odd-even.cu](base/cuda-odd-even.cu)
- [hpc.h](../../include/hpc.h)
