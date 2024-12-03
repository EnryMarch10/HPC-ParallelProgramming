# Dense matrix-matric product using vector datatypes

The file [simd-matmul.c](base/simd-matmul.c) contains a serial version of the dense matrix-matrix product of two square matrices
$p, q$, $r=p \times q$.
Both a "plain" and cache-efficient version are provided; the cache-efficient program transposes $q$ so that the product can be
computed by accessing both $p$ and $q^T$ row-wise.

The cache-efficient matrix-matrix product can be modified to take advantage of SIMD instructions, since it essentially computes a
number of dot products between rows of $p$ and rows of $q^T$.
Indeed, the body of function `scalar_matmul_tr()`

```C
for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++) {
		double s = 0.0;
		for (int k = 0; k < n; k++) {
			s += p[i * n + k] * qT[j * n + k];
		}
		r[i * n + j] = s;
	}
}
```

computes the dot product of two arrays of length $n$ that are stored at memory addresses
$(p + i \times n)$ and $(\mathit{qT} + j \times n)$, respectively.

Your goal is to use SIMD instructions to compute the dot product above using _vector datatypes_ provided by the GCC compiler.
The program guarantees that the array length $n$ is an integer multiple of the SIMD register length, and that all data are
suitably aligned in memory.

This exercise uses the `double` data type; it is therefore necessary to define a vector datatype `v2d` of length 16 Bytes
containing two doubles, using the declaration:

```C
typedef double v2d __attribute__((vector_size(16)));
#define VLEN (sizeof(v2d) / sizeof(double))
```

The server `isi-raptor03` supports the AVX2 instruction set, and therefore has SIMD registers of width 256 bits = 32 Bytes.
You might want to make use of a wider datatype `v4d` containing 4 doubles instead of two.

To compile:

```shell
gcc -march=native -O2 -std=c99 -Wall -Wpedantic simd-matmul.c -o simd-matmul
```

To execute:

```shell
./simd-matmul [matrix size]
```

Example:

```shell
./simd-matmul 1024
```

## Files

- [simd-matmul.c](base/simd-matmul.c)
- [hpc.h](../../include/hpc.h)
