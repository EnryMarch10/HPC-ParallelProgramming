SHELL = /bin/bash

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -std=c99 -Wall -Wpedantic -Werror -O0 -fopenmp

# Target executable
TARGET = omp-cat-map

# Source files
SRCS = $(TARGET).c pgm.c
# Object files
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

# Build the target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: clean all
