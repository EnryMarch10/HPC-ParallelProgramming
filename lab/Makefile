SHELL = /bin/bash

LAB1 := lab1
LAB2 := lab2
LAB3 := lab3

LABS := $(LAB1) $(LAB2) $(LAB3)

DIR_LAB1 := 01_OpenMP-1
DIR_LAB2 := 02_OpenMP-2
DIR_LAB3 := 03_MPI-1

all: $(LABS)

help:
	@echo
	@echo "Available targets:"
	@echo
	@echo "     help    prints this message"
	@echo "     lab1    compile all lab1 programs (OpenMP-1)"
	@echo "     lab2    compile all lab2 programs (OpenMP-2)"
	@echo "     lab3    compile all lab3 programs (MPI-1)"
	@echo "      all    compile everything (default)"
	@echo "    clean    cleanup temporary files"
	@echo

$(LAB1):
	@echo "*** TARGET - $(LAB1) ***"
	for TARGET in `find ./$(DIR_LAB1) -mindepth 1 -maxdepth 1 -type d -print`; do\
		if ! make -C "$$TARGET"; then\
			exit $$?;\
		fi;\
	done

$(LAB2):
	@echo "*** TARGET - $(LAB2) ***"
	for TARGET in `find ./$(DIR_LAB2) -mindepth 1 -maxdepth 1 -type d -print`; do\
		if ! make -C "$$TARGET"; then\
			exit $$?;\
		fi;\
	done

$(LAB3):
	@echo "*** TARGET - $(LAB3) ***"
	for TARGET in `find ./$(DIR_LAB3) -mindepth 1 -maxdepth 1 -type d -print`; do\
		if ! make -C "$$TARGET"; then\
			exit $$?;\
		fi;\
	done

clean:
	@echo "*** TARGET - $(LAB1) ***"
	for TARGET in `find ./$(DIR_LAB1) -mindepth 1 -maxdepth 1 -type d -print`; do\
		if ! make -C "$$TARGET" clean; then\
			exit $$?;\
		fi;\
	done
	@echo "*** TARGET - $(LAB2) ***"
	for TARGET in `find ./$(DIR_LAB2) -mindepth 1 -maxdepth 1 -type d -print`; do\
		if ! make -C "$$TARGET" clean; then\
			exit $$?;\
		fi;\
	done
	@echo "*** TARGET - $(LAB3) ***"
	for TARGET in `find ./$(DIR_LAB3) -mindepth 1 -maxdepth 1 -type d -print`; do\
		if ! make -C "$$TARGET" clean; then\
			exit $$?;\
		fi;\
	done

.PHONY: all help $(LABS) clean
