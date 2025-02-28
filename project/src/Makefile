SHELL := /bin/bash

## BUILD dir

DIR_BUILD := build
DIR_BIN := $(DIR_BUILD)/bin
DIR_OBJ := $(DIR_BUILD)/obj

DIRS := $(DIR_BIN) $(DIR_OBJ) $(DIR_BUILD)

## PROGRAMS

INPUT := input
SERIAL := serial
OPENMP := omp
MPI := mpi
CUDA := cuda

TARGETS := $(INPUT) $(SERIAL) $(OPENMP) $(MPI) $(CUDA)

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
	@echo "   input    compile worst case input generator program"
	@echo "   files    generate all input files"
	@echo "  serial    compile serial program"
	@echo "     omp    compile parallel OpenMP program"
	@echo "     mpi    compile MPI program"
	@echo "    cuda    compile CUDA program"
	@echo "     all    compile everything (default)"
	@echo "   clean    cleanup temporary files"
	@echo "   purge    brutally removes build/* with all contents"
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

purge:
	-\rm -rf $(DIR_BUILD)

## INPUT

N_POINTS := 100000
DATA_DIR := input

files: \
	$(DATA_DIR)/circle-N1000-D2.in \
	$(DATA_DIR)/test1-N$(N_POINTS)-D3.in \
	$(DATA_DIR)/test2-N$(N_POINTS)-D4.in \
	$(DATA_DIR)/test3-N$(N_POINTS)-D10.in \
	$(DATA_DIR)/test4-N$(N_POINTS)-D8.in \
	$(DATA_DIR)/test5-N$(N_POINTS)-D20.in \
	$(DATA_DIR)/test6-N$(N_POINTS)-D50.in \
	$(DATA_DIR)/test7-N$(N_POINTS)-D200.in \
	$(DATA_DIR)/worst-N$(N_POINTS)-D10.in

# a small 2d circle
$(DATA_DIR)/circle-N1000-D2.in:
	@mkdir -p $(DATA_DIR)
	rbox 1000 s D2 W1 > $@

# surface of a 3d square
$(DATA_DIR)/test1-N$(N_POINTS)-D3.in:
	@mkdir -p $(DATA_DIR)
	rbox $(N_POINTS) W0 > $@

# surface of a 4d sphere
$(DATA_DIR)/test2-N$(N_POINTS)-D4.in:
	@mkdir -p $(DATA_DIR)
	rbox $(N_POINTS) s D4 > $@

# 10d diamond
$(DATA_DIR)/test3-N$(N_POINTS)-D10.in:
	@mkdir -p $(DATA_DIR)
	rbox $(N_POINTS) d D10 > $@

# 8d symplex
$(DATA_DIR)/test4-N$(N_POINTS)-D8.in:
	@mkdir -p $(DATA_DIR)
	rbox $(N_POINTS) y D8 > $@

# 20d sphere
$(DATA_DIR)/test5-N$(N_POINTS)-D20.in:
	@mkdir -p $(DATA_DIR)
	rbox $(N_POINTS) s D20 > $@

# 50d diamond
$(DATA_DIR)/test6-N$(N_POINTS)-D50.in:
	@mkdir -p $(DATA_DIR)
	rbox $(N_POINTS) d D50 > $@

# 200d diamond
$(DATA_DIR)/test7-N$(N_POINTS)-D200.in:
	@mkdir -p $(DATA_DIR)
	rbox $(N_POINTS) d D200 > $@

# worst-case scenario
$(DATA_DIR)/worst-N$(N_POINTS)-D10.in: $(TARGET_INPUT)
	@mkdir -p $(DATA_DIR)
	$(DIR_BIN)/$(EXE_INPUT) $(N_POINTS) 10 > $@

## Special targets

.PHONY: files all help clean purge $(TARGETS) $(EXE)
