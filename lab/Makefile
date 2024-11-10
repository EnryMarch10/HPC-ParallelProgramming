SHELL = /bin/bash

LAB1 := lab1
LAB2 := lab2
LABS := $(LAB1) $(LAB2)

all: $(LABS)

help:
	@echo
	@echo "Available targets:"
	@echo
	@echo "     help    prints this message"
	@echo "     lab1    compile all lab1 programs (OpenMP-1)"
	@echo "     lab2    compile all lab2 programs (OpenMP-2)"
	@echo "      all    compile everything (default)"
	@echo "    clean    cleanup temporary files"
	@echo

$(LAB1):
	@echo "*** TARGET - $(LAB1) ***"
	for TARGET in `find ./01_OpenMP-1 -mindepth 1 -maxdepth 1 -type d -print`; do\
		if ! make -C "$$TARGET"; then\
			exit $$?;\
		fi;\
	done

$(LAB2):
	@echo "*** TARGET - $(LAB2) ***"
	@echo "Error! This command is not implemented yet" >&2

# for TARGET in `find ./01_OpenMP-1 -mindepth 1 -maxdepth 1 -type d -print`; do\
# 	if ! make -C "$$TARGET" clean; then\
# 		exit $$?;\
# 	fi;\
# done

clean:
	for TARGET in `find ./01_OpenMP-1 -mindepth 1 -maxdepth 1 -type d -print`; do\
		if ! make -C "$$TARGET" clean; then\
			exit $$?;\
		fi;\
	done

.PHONY: all help $(LABS) clean
