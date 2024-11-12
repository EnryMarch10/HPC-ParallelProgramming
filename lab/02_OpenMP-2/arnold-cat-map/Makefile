SHELL = /bin/bash

SERIAL := serial
PARALLEL := parallel
SOLUTION := solution

TARGETS := $(SERIAL) $(PARALLEL) $(SOLUTION)

DIR_BUILD := build
DIR_BIN := $(DIR_BUILD)/bin
DIR_OBJ := $(DIR_BUILD)/obj

DIRS := $(DIR_BIN) $(DIR_OBJ) $(DIR_BUILD)

SOURCE_1_SERIAL := cat-map
SOURCE_1_PARALLEL := omp-$(SOURCE_1_SERIAL)
SOURCE_1_PARALLEL_DIP_1 = pgm
SOURCE_1_PARALLEL_ALL = $(SOURCE_1_PARALLEL) $(SOURCE_1_PARALLEL_DIP_1)
SOURCE_1_SOLUTION := sol-$(SOURCE_1_PARALLEL)

SOURCE_2_SERIAL := cat-map-rectime
SOURCE_2_PARALLEL := omp-$(SOURCE_2_SERIAL)
SOURCE_2_PARALLEL_1 := $(SOURCE_2_PARALLEL)-simple
SOURCE_2_PARALLEL_2 := $(SOURCE_2_PARALLEL)-sta-sta
SOURCE_2_PARALLEL_3 := $(SOURCE_2_PARALLEL)-dyn-dyn
SOURCE_2_PARALLEL_4 := $(SOURCE_2_PARALLEL)-sta-dyn
SOURCE_2_PARALLEL_5 := $(SOURCE_2_PARALLEL)-stachunk-dyn

SOURCES := $(SOURCE_1_SERIAL) $(SOURCE_1_PARALLEL_ALL) $(SOURCE_1_SOLUTION) \
           $(SOURCE_2_SERIAL) $(SOURCE_2_PARALLEL_1) $(SOURCE_2_PARALLEL_2) \
           $(SOURCE_2_PARALLEL_3) $(SOURCE_2_PARALLEL_4) $(SOURCE_2_PARALLEL_5)

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

### SERIAL ###
$(SERIAL): $(SOURCE_1_SERIAL) $(SOURCE_2_SERIAL)

# Temporary, source code should be changed so as to not depend on OpenMP
$(SOURCE_1_SERIAL): CFLAGS += -fopenmp
$(SOURCE_1_SERIAL): $(DIR_BIN)/$(SOURCE_1_SERIAL)

$(DIR_BIN)/$(SOURCE_1_SERIAL): $(DIR_OBJ)/$(SOURCE_1_SERIAL).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_1_SERIAL).o: $(SERIAL)/$(SOURCE_1_SERIAL).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

$(SOURCE_2_SERIAL): $(DIR_BIN)/$(SOURCE_2_SERIAL)

$(DIR_BIN)/$(SOURCE_2_SERIAL): $(DIR_OBJ)/$(SOURCE_2_SERIAL).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_2_SERIAL).o: $(SERIAL)/$(SOURCE_2_SERIAL).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

### PARALLEL ###
$(PARALLEL): $(SOURCE_1_PARALLEL) $(SOURCE_2_PARALLEL_1) $(SOURCE_2_PARALLEL_2) $(SOURCE_2_PARALLEL_3) $(SOURCE_2_PARALLEL_4) $(SOURCE_2_PARALLEL_5)

# parallel 1
$(SOURCE_1_PARALLEL): CFLAGS += -fopenmp
$(SOURCE_1_PARALLEL): $(DIR_BIN)/$(SOURCE_1_PARALLEL)

$(DIR_BIN)/$(SOURCE_1_PARALLEL): $(addsuffix .o, $(addprefix $(DIR_OBJ)/, $(SOURCE_1_PARALLEL_ALL)))
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_1_PARALLEL).o: $(PARALLEL)/$(SOURCE_1_PARALLEL).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

$(DIR_OBJ)/$(SOURCE_1_PARALLEL_DIP_1).o: $(PARALLEL)/$(SOURCE_1_PARALLEL_DIP_1).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

# parallel 2 - version 1
$(SOURCE_2_PARALLEL_1): CFLAGS += -fopenmp
$(SOURCE_2_PARALLEL_1): $(DIR_BIN)/$(SOURCE_2_PARALLEL_1)

$(DIR_BIN)/$(SOURCE_2_PARALLEL_1): $(DIR_OBJ)/$(SOURCE_2_PARALLEL_1).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_2_PARALLEL_1).o: $(PARALLEL)/$(SOURCE_2_PARALLEL_1).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

# parallel 2 - version 2
$(SOURCE_2_PARALLEL_2): CFLAGS += -fopenmp
$(SOURCE_2_PARALLEL_2): $(DIR_BIN)/$(SOURCE_2_PARALLEL_2)

$(DIR_BIN)/$(SOURCE_2_PARALLEL_2): $(DIR_OBJ)/$(SOURCE_2_PARALLEL_2).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_2_PARALLEL_2).o: $(PARALLEL)/$(SOURCE_2_PARALLEL_2).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

# parallel 2 - version 3
$(SOURCE_2_PARALLEL_3): CFLAGS += -fopenmp
$(SOURCE_2_PARALLEL_3): $(DIR_BIN)/$(SOURCE_2_PARALLEL_3)

$(DIR_BIN)/$(SOURCE_2_PARALLEL_3): $(DIR_OBJ)/$(SOURCE_2_PARALLEL_3).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_2_PARALLEL_3).o: $(PARALLEL)/$(SOURCE_2_PARALLEL_3).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

# parallel 2 - version 4
$(SOURCE_2_PARALLEL_4): CFLAGS += -fopenmp
$(SOURCE_2_PARALLEL_4): $(DIR_BIN)/$(SOURCE_2_PARALLEL_4)

$(DIR_BIN)/$(SOURCE_2_PARALLEL_4): $(DIR_OBJ)/$(SOURCE_2_PARALLEL_4).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_2_PARALLEL_4).o: $(PARALLEL)/$(SOURCE_2_PARALLEL_4).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

# parallel 2 - version 5
$(SOURCE_2_PARALLEL_5): CFLAGS += -fopenmp
$(SOURCE_2_PARALLEL_5): $(DIR_BIN)/$(SOURCE_2_PARALLEL_5)

$(DIR_BIN)/$(SOURCE_2_PARALLEL_5): $(DIR_OBJ)/$(SOURCE_2_PARALLEL_5).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_2_PARALLEL_5).o: $(PARALLEL)/$(SOURCE_2_PARALLEL_5).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

### SOLUTION ###
$(SOLUTION): $(SOURCE_1_SOLUTION)

$(SOURCE_1_SOLUTION): CFLAGS += -fopenmp
$(SOURCE_1_SOLUTION): $(DIR_BIN)/$(SOURCE_1_SOLUTION)

$(DIR_BIN)/$(SOURCE_1_SOLUTION): $(DIR_OBJ)/$(SOURCE_1_SOLUTION).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(SOURCE_1_SOLUTION).o: $(SOLUTION)/$(SOURCE_1_SOLUTION).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	-\rm -f $(addprefix $(DIR_BIN)/, $(SOURCES)) \
	        $(addprefix $(DIR_OBJ)/, $(addsuffix .o, $(SOURCES)))
	-\rm -df $(DIRS)

.PHONY: all $(TARGETS) help clean
