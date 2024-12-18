## Written by Enrico Marchionni

SHELL := /bin/bash

INPUT := input
SERIAL := serial
OPENMP := omp
MPI := mpi
CUDA := cuda

TARGETS := $(INPUT) $(SERIAL) $(OPENMP) $(MPI) $(CUDA)

DIR_BUILD := build
DIR_BIN := $(DIR_BUILD)/bin
DIR_OBJ := $(DIR_BUILD)/obj

DIRS := $(DIR_BIN) $(DIR_OBJ) $(DIR_BUILD)

EXE_INPUT := inputgen
TARGET_INPUT := $(addprefix $(DIR_BIN)/, $(EXE_INPUT))
EXE_SERIAL := skyline
TARGET_SERIAL := $(addprefix $(DIR_BIN)/, $(EXE_SERIAL))
EXE_OMP := $(basename $(wildcard omp-*.c))
TARGET_OMP := $(addprefix $(DIR_BIN)/, $(EXE_OMP))
EXE_MPI := $(basename $(wildcard mpi-*.c))
TARGET_MPI := $(addprefix $(DIR_BIN)/, $(EXE_MPI))
EXE_CUDA := $(basename $(wildcard cuda-*.cu))
TARGET_CUDA := $(addprefix $(DIR_BIN)/, $(EXE_CUDA))

EXE := $(EXE_INPUT) $(EXE_SERIAL) $(EXE_OMP) $(EXE_MPI) $(EXE_CUDA)

# C
CC = gcc
CFLAGS += -std=c99 -Wall -Wpedantic -Werror -O2 -D_XOPEN_SOURCE=600
LDLIBS += -lm

# CUDA
NVCC := nvcc
NVCFLAGS += -Wno-deprecated-gpu-targets
NVLDLIBS += -lm

all: $(TARGETS)

$(INPUT): $(EXE_INPUT)
$(EXE_INPUT): $(TARGET_INPUT)

$(SERIAL): $(EXE_SERIAL)
$(EXE_SERIAL): $(TARGET_SERIAL)

$(OPENMP): $(EXE_OMP)
$(EXE_OMP): $(TARGET_OMP)
$(TARGET_OMP): CFLAGS += -fopenmp

$(MPI): $(EXE_MPI)
$(EXE_MPI): $(TARGET_MPI)
$(TARGET_MPI): CC = mpicc

$(CUDA): $(EXE_CUDA)
$(EXE_CUDA): $(TARGET_CUDA)

help:
	@echo
	@echo "Available targets:"
	@echo
	@echo "    help    prints this message"
	@echo "  serial    compile serial program"
	@echo "     omp    compile parallel OpenMP program"
	@echo "     mpi    compile MPI program"
	@echo "    cuda    compile CUDA program"
	@echo "     all    compile everything (default)"
	@echo "   clean    cleanup temporary files"
	@echo

.PRECIOUS: $(DIR_OBJ)/%.o

# C, OpenMP and MPI

$(DIR_BIN)/%: $(DIR_OBJ)/%.o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(DIR_OBJ)/%.o: %.c hpc.h
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $< -o $@ $(LDLIBS)

# CUDA

$(DIR_BIN)/%: %.cu
	@mkdir -p $(DIR_BIN)
	$(NVCC) $(NVCFLAGS) $^ -o $@ $(NVLDLIBS)

# INPUT

$(DIR_BIN)/$(EXE_INPUT): $(DIR_OBJ)/$(EXE_INPUT).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(EXE_INPUT).o: $(EXE_INPUT).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	-\rm -f $(addprefix $(DIR_OBJ)/, $(addsuffix .o, $(EXE))) $(addprefix $(DIR_BIN)/, $(EXE))
	-\rm -df $(DIRS)

.PHONY: all help clean $(TARGETS) $(EXE)
