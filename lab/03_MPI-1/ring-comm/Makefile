SHELL = /bin/bash

PARALLEL := parallel
SOLUTION := solution

TARGETS := $(PARALLEL) $(SOLUTION)

DIR_BUILD := build
DIR_BIN := $(DIR_BUILD)/bin
DIR_OBJ := $(DIR_BUILD)/obj

DIRS := $(DIR_BIN) $(DIR_OBJ) $(DIR_BUILD)

SOURCE := ring
SOURCE_PARALLEL := mpi-$(SOURCE)
SOURCE_SOLUTION := sol-$(SOURCE_PARALLEL)

SOURCES := $(SOURCE_PARALLEL) $(SOURCE_SOLUTION)

CC := mpicc
CFLAGS = -std=c99 -Wall -Wpedantic -Werror

all: $(TARGETS)

help:
	@echo
	@echo "Available targets:"
	@echo
	@echo "     help    prints this message"
	@echo " parallel    compile parallel OpenMP program"
	@echo " solution    compile solution program"
	@echo "      all    compile everything (default)"
	@echo "    clean    cleanup temporary files"
	@echo

### PARALLEL ###
$(PARALLEL): $(SOURCE_PARALLEL)

$(SOURCE_PARALLEL): $(DIR_BIN)/$(SOURCE_PARALLEL)

$(DIR_BIN)/$(SOURCE_PARALLEL): $(DIR_OBJ)/$(SOURCE_PARALLEL).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_PARALLEL).o: $(PARALLEL)/$(SOURCE_PARALLEL).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

### SOLUTION ###
$(SOLUTION): $(SOURCE_SOLUTION)

$(SOURCE_SOLUTION): $(DIR_BIN)/$(SOURCE_SOLUTION)

$(DIR_BIN)/$(SOURCE_SOLUTION): $(DIR_OBJ)/$(SOURCE_SOLUTION).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_SOLUTION).o: $(SOLUTION)/$(SOURCE_SOLUTION).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	-\rm -f $(addprefix $(DIR_BIN)/, $(SOURCES)) \
	       $(addprefix $(DIR_OBJ)/, $(addsuffix .o, $(SOURCES)))
	-\rm -df $(DIRS)

.PHONY: all $(TARGETS) help clean
