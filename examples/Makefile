## This Makefile compiles the source programs discussed during the
## High Performance Computing class, 2024/2025, laurea degree in
## computer science and engineering, University of Bologna, Italy.
##
## To compile MPI targets you need the `mpicc` compiler wrapper.
## To compile CUDA targets you need the `nvcc` compiler.
## To compile OpenCL targets you need a functional OpenCl installation.
##
## Last modified 2024-10-14
## Moreno Marzolla <https://www.moreno.marzolla.name/>
##
## `make`	generates all targets
## `make help`	shows the available targets

DIR_BUILD := build
DIR_BIN := $(DIR_BUILD)/bin
DIR_OBJ := $(DIR_BUILD)/obj
DIR_ASM := $(DIR_BUILD)/asm
DIR_LST := $(DIR_BUILD)/lst

DIRS := $(DIR_BIN) $(DIR_OBJ) $(DIR_ASM) $(DIR_LST) $(DIR_BUILD)

EXE_OMP := $(basename $(filter-out omp-bug1.c omp-bug2.c, $(wildcard omp-*.c)))
EXE_MPI := $(basename $(wildcard mpi-*.c))
EXE_SERIAL := trap odd-even prefix-sum matmul-plain matmul-transpose game-of-life reduction vla-demo game-of-life-new matmul-test matmul_test.class
EXE_SIMD := $(basename $(wildcard simd-*.c)) simd-vsum-vector.lst simd-vsum-auto.lst
EXE_CUDA := $(basename $(wildcard *.cu))
EXE_OPENCL := $(basename $(wildcard opencl-*.c))
EXE := $(addprefix $(DIR_BIN)/, $(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_SIMD) $(EXE_OPENCL) $(EXE_CUDA))
DISTFILES := $(wildcard *.cu *.c *.h *.cl *.sh *.ods *.py *.java) Makefile README cat-1344.pgm mandelbrot-set-demo.ggb
DISTNAME := examples
CFLAGS += -std=c99 -Wall -Wpedantic # -Werror
MPICC ?= mpicc
NVCC ?= nvcc
NVCFLAGS +=

all: mpi omp serial simd opencl cuda

help:
	@echo
	@echo "Available targets:"
	@echo
	@echo "     help    prints this message"
	@echo "   serial    compile all serial programs"
	@echo "      omp    compile all OpenMP programs"
	@echo "      mpi    compile all MPI programs"
	@echo "   opencl    compile all OpenLC programs"
	@echo "     cuda    compil3 all CUDA programs"
	@echo "     simd    compile all SIMD programs"
	@echo "      all    compile everything (default)"
	@echo "    clean    cleanup temporary files"
	@echo
	@echo "To install MPI on Debian/Ubuntu:"
	@echo
	@echo "    sudo apt install mpi-default-bin mpi-default-dev"
	@echo
	@echo "To install OpenCl on Debian/Ubuntu:"
	@echo
	@echo "    sudo apt install pocl-opencl-icd    # if you have no GPU"
	@echo "    sudo apt install beignet-opencl-icd # if you have Intel HD GPUs"
	@echo "    sudo apt install libclc-amdgcn      # if you have AMD GPPUs"
	@echo "    sudo apt install nvidia-opencl-icd  # if you have NVidia GPUs"
	@echo
	@echo "If you have an NVidia GPU, you can download the CUDA SDK here"
	@echo
	@echo "    https://developer.nvidia.com/cuda-downloads"
	@echo

dist: $(DISTNAME).zip

$(DISTNAME).zip: $(DISTFILES)
	rm -r -f $(DISTNAME)
	mkdir $(DISTNAME)
	cd $(DISTNAME)
	ln $(DISTFILES) $(DISTNAME)
	zip -r $(DISTNAME).zip $(DISTNAME)

gfx.o: gfx.c gfx.h

omp-mandelbrot: omp-mandelbrot.o gfx.o
omp-mandelbrot: LDLIBS += -lX11
omp-mandelbrot: LDFLAGS += -fopenmp
matmul-plain: LDLIBS += -lrt
matmul-plain: matmul.c
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $(DIR_BIN)/$@

matmul-transpose: CFLAGS += -DTRANSPOSE
matmul-transpose: matmul.c
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $< -o $(DIR_BIN)/$@

matmul-test: CFLAGS += -fopenmp -mavx2 -mfma -O2

matmul_test.class: matmul_test.java
	@mkdir -p $(DIR_BIN)
	javac -d $(DIR_BIN) $<

$(EXE_OMP): CFLAGS += -fopenmp
$(EXE_OMP): LDLIBS += -lrt
omp: $(EXE_OMP)

$(EXE_MPI): CC=$(MPICC)
mpi: $(EXE_MPI)

mpi-vecsum mpi-vecsum2: LDLIBS += -lm

serial: $(EXE_SERIAL)

$(EXE_SIMD): CFLAGS += -march=native
simd: $(EXE_SIMD)

simd-vsum-vector: CFLAGS += -O2 -lm

simd-vsum-vector.s: simd-vsum-vector.c
	@mkdir -p $(DIR_ASM)
	gcc -S -c -fverbose-asm -g -O2 -march=native -funsafe-math-optimizations -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized $< -o $(DIR_ASM)/$@

simd-vsum-intrinsics: CFLAGS += -O2 -lm

simd-vsum-auto: CFLAGS += -lm -O2 -funsafe-math-optimizations -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized

simd-vsum-auto.s: simd-vsum-auto.c
	@mkdir -p $(DIR_ASM)
	gcc -S -c -fverbose-asm -g -O2 -march=native -funsafe-math-optimizations -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized $< -o $(DIR_ASM)/$@

$(EXE_OPENCL): CFLAGS += simpleCL.c
$(EXE_OPENCL): LDLIBS += -lOpenCL
opencl: $(EXE_OPENCL)

opencl-rotate opencl-matmul: LDLIBS += -lm

cuda: $(EXE_CUDA)

cuda-rotate cuda-matmul: NVCFLAGS += -lm

clean:
	\rm -f $(EXE) $(DIR_OBJ)/omp-mandelbrot.o $(DIR_OBJ)/gfx.o $(DIR_LST)/simd-vsum-vector.lst $(DIR_LST)/simd-vsum-auto.lst \
	$(DIR_ASM)/simd-vsum-vector.s $(DIR_ASM)/simd-vsum-auto.s
	\rmdir --ignore-fail-on-non-empty $(DIRS)

distclean: clean
	\rm -r -f $(DISTNAME) $(DISTNAME).zip

.PHONY: all clean distclean

# Implicit Rules - changes

%.s:

%.lst: %.s
	@mkdir -p $(DIR_LST)
	as -alhnd $(addprefix $(DIR_ASM)/, $<) -o $(DIR_LST)/$*.out > $(DIR_LST)/$@

%.o:

%: %.o
	@mkdir -p $(DIR_BIN)
	$(LINK.o) $(addprefix $(DIR_OBJ)/, $^) $(LOADLIBES) $(LDLIBS) -o $(DIR_BIN)/$@

%.c:

%: %.c
	@mkdir -p $(DIR_BIN)
	$(LINK.c) $^ $(LOADLIBES) $(LDLIBS) -o $(DIR_BIN)/$@

%.o: %.c
	@mkdir -p $(DIR_OBJ)
	$(COMPILE.c) -o $(DIR_OBJ)/$@ $<

%.cu:

%: %.cu
	@mkdir -p $(DIR_BIN)
	$(NVCC) $(NVCFLAGS) $^ -o $(DIR_BIN)/$@
