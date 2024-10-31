#!/bin/env python3

# Code from the MIT course "Performance engineering of software systems"
#
# Modified for Python3 by Moreno Marzolla
# Last modified 2024-10-16 by Moreno Marzolla

import sys, random
from time import time

n = 4096

p = [[random.random() for row in range(n)]
     for col in range(n)]
q = [[random.random() for row in range(n)]
     for col in range(n)]
r = [[0 for row in range(n)]
     for col in range(n)]

print("Matrix-Matrix multiplication (Python) %d x %d\n" % (n, n));

start = time()
for i in range(n):
    for j in range(n):
        for k in range(n):
            r[i][j] += p[i][k] * q[k][j]
elapsed = time() - start
Gflops = 2.0 * ((n/1000) ** 3) / elapsed
print("      Time\t    Gflops")
print("----------\t----------")
print("%10.3f\t%10.3f" % (elapsed, Gflops))
