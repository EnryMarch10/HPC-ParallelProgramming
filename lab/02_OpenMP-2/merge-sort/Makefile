SHELL = /bin/bash

SERIAL := serial
PARALLEL := parallel
SOLUTION := solution

TARGETS := $(SERIAL) $(PARALLEL) $(SOLUTION)

DIR_BUILD := build
DIR_BIN := $(DIR_BUILD)/bin
DIR_OBJ := $(DIR_BUILD)/obj

DIRS := $(DIR_BIN) $(DIR_OBJ) $(DIR_BUILD)

SOURCE_SERIAL := merge-sort
SOURCE_PARALLEL := omp-$(SOURCE_SERIAL)
SOURCE_SOLUTION := sol-$(SOURCE_PARALLEL)

SOURCES := $(SOURCE_SERIAL) $(SOURCE_PARALLEL) $(SOURCE_SOLUTION)

CC := gcc
CFLAGS = -std=c99 -Wall -Wpedantic -Werror

all: $(TARGETS)

help:
	@echo
	@echo "Available targets:"
	@echo
	@echo "     help    prints this message"
	@echo "   serial    compile serial program"
	@echo " parallel    compile parallel OpenMP program"
	@echo " solution    compile solution program"
	@echo "      all    compile everything (default)"
	@echo "    clean    cleanup temporary files"
	@echo

$(SERIAL): $(SOURCE_SERIAL)

$(SOURCE_SERIAL): $(DIR_BIN)/$(SOURCE_SERIAL)

$(DIR_BIN)/$(SOURCE_SERIAL): $(DIR_OBJ)/$(SOURCE_SERIAL).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_SERIAL).o: $(SERIAL)/$(SOURCE_SERIAL).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

$(PARALLEL): $(SOURCE_PARALLEL)

$(SOURCE_PARALLEL): CFLAGS += -fopenmp
$(SOURCE_PARALLEL): $(DIR_BIN)/$(SOURCE_PARALLEL)

$(DIR_BIN)/$(SOURCE_PARALLEL): $(DIR_OBJ)/$(SOURCE_PARALLEL).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_PARALLEL).o: $(PARALLEL)/$(SOURCE_PARALLEL).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

$(SOLUTION): $(SOURCE_SOLUTION)

$(SOURCE_SOLUTION): CFLAGS += -fopenmp
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
