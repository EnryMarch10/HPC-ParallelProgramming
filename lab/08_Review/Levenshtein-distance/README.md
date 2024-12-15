# OpenMP - Levenshtein’s edit distance

The file [omp-levenshtein.c](base/omp-levenshtein.c) contains a serial implementation of
[Levenshtein's algorithm](https://en.wikipedia.org/wiki/Levenshtein_distance) for computing the _edit distance_ between two
strings.
Levenshtein's edit distance is a measure of similarity, and is related to the minimum number of _edit operations_ that are
required to transform one string into another.
Several types of edit operations have been considered in the literature; in this program we consider insertion, deletion and
replacement of single characters while scanning the string from left to right.

Levenshtein's distance can be computed using _dynamic programming_.
To solve this exercise you are not required to know the details; for the sake of completeness (and for those who are interested),
a brief description of the algorithm is provided below.

Let $s$ and $t$ be two strings of lengths $n \geq 0, m \geq 0$ respectively.
Let $L[i][j]$ be the edit distance between the prefix $s[0 \ldots i-1]$ of length $i$ and $t[0 \ldots j-1]$ of length $j$,
$i=0, \ldots, n$, $j = 0, \ldots, m$.
In other words, $L[i][j]$ is the minimum number of edit operations that are required to transform the first $i$ characters of $s$
into the first $j$ characters of $t$.

The base case arises when one of the prefixes is empty, i.e., $i=0$ or $j=0$:

- If $i=0$ then the first prefix is empty, so to transform an empty string into $t[0 \ldots j-1]$ we need $j$ insert operations:
  $L[0][j] = j$.

- If $j=0$ then the second prefix is empty, so to transform $s[0 \ldots i-1]$ into the empty string we need $i$ removal
  operations: $L[i][0] = i$.

If both $i$ and $j$ are nonzero, we  have three possibilities (see Fig. 1):

1. Delete the last character of $s[0 \ldots i-1]$ and transform $s[0 \ldots i-2]$ into $t[0 \ldots j-1]$. Cost: $1 + L[i-1][j]$
   (one delete operation, plus the cost of transforming $s[i-2]$ into $t[j-1]$).

2. Delete the last character of $t[0 \ldots j-1]$ and transform $s[0 \ldots i-1]$ into $t[0 \ldots j-2]$. Cost: $1 + L[i][j-1]$.

3. Depending on the last characters of the prefixes
   of $s$ and $t$:

    1. If the last characters are the same ($s[i-1] = t[j+1]   $), then we may keep the last characters and transform
       $s[0 \ldots i-2]$ into $t[0 \ldots j-2]$. Cost: $L[i-1][j-1]$.

    2. If the last characters are different (i.e., $s[i-1] \neq t[i-1]$), we can replace $s[i-1]$ with $t[j-1]$, and transform
       $s[0 \ldots i-2]$ into $t[0 \ldots j-2]$. Cost: $1 + L[i-1][j-1]$.

![Figure 1: Computation of $L[i][j]$](img/omp-levenshtein.png)

We choose the alternative that minimizes the cost, so we can summarize the cases above with the following expression:

$$
  L[i][j] = \begin{cases}
  j & \text{if $i = 0, j > 0$} \\
  i & \text{if $i > 0, j = 0$} \\
  1 + \min\{L[i][j-1], L[i-1][j], L[i-1][j-1] + 1_{s[i-1] = t[j-1]}\}& \text{if $i>0, j>0$}
  \end{cases}
$$

where $1_P$ is the _indicator function_ of predicate $P$, i.e., a function whose value is 1 iff $P$ is true, 0 otherwise.

The core of the algorithm is the computation of the entries of matrix $L$ of size $(n+1) \times (m+1)$; the equation above shows
that the matrix can be filled using two nested loops, and is based on a _three-point stencil_ since the value of each element
depends of the value above, on the left, and on the upper left corner.

Unfortunately, it is not possible to apply an `omp parallel for` directive to either loops due to loop-carried dependencies.
However, we can rewrite the loops so that the matrix is filled diagonally through a _wavefront computation_.
The computation of the values on the diagonal can indeed be computed in parallel since they have no inter-dependencies.

The wavefront computation can be implemented as follows:

```C
for (int slice = 0; slice < n + m - 1; slice++) {
    const int z1 = slice < m ? 0 : slice - m + 1;
    const int z2 = slice < n ? 0 : slice - n + 1;
    for (int ii = slice - z2; ii >= z1; ii--) {
        const int jj = slice - ii;
        const int i = ii + 1;
        const int j = jj + 1;
        L[i][j] = min3(L[i - 1][j] + 1,
                       L[i][j - 1] + 1,
                       L[i - 1][j - 1] + (s[i - 1] != t[j - 1]));
    }
}
```

and the inner loop can be parallelized.

Compile with:

```shell
gcc -std=c99 -Wall -Wpedantic -fopenmp omp-levenshtein.c -o omp-levenshtein
```

Run with:

```shell
./omp-levenshtein str1 str2
```

Example:

```shell
./omp-levenshtein "this is a test" "that test is different"
```

## Files

- [omp-levenshtein.c](base/omp-levenshtein.c)
