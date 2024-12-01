# Matrix-matrix addition

The program [cuda-matsum.cu](base/cuda-matsum.cu) computes the sum of two square matrices of size $N \times N$ using the CPU.
Modify the program to use the GPU.
You must modify the function `matsum()` in such a way that the new version is transparent to
the caller, i.e., the caller is not aware whether the computation happens on the CPU or the GPU.
To this aim, function `matsum()` should:

- allocate memory on the device to store copies of $p, q, r$

- copy $p, q$ from the _host_ to the _device_

- execute a kernel that computes the sum $p + q$

- copy the result from the _device_ back to the _host_

- free up device memory

The program must work with any value of the matrix size $N$, even if it is not an integer multiple of the CUDA block size.
Note that there is no need to use shared memory: why?

To compile:

```shell
nvcc cuda-matsum.cu -o cuda-matsum -lm
```

To execute:

```shell
./cuda-matsum [N]
```

Example:

```shell
./cuda-matsum 1024
```

## Files

- [cuda-matsum.cu](base/cuda-matsum.cu)
- [hpc.h](../../include/hpc.h)
