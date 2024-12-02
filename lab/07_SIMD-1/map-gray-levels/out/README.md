# Analyzing assembly instructions

This works for linux **gcc** compiler.

Assembly file can be created with specific flag (`-S` and `-fverbose-asm` to add comments):

```shell
gcc -S -fverbose-asm -std=c99 -Wall -Wpedantic -march=native simd-map-levels.c -o simd-map-levels.s -I../../../include
```

> **-S**
>
> Stop after the stage of compilation proper; do not assemble.
> The output is in the form of an assembler code file for each non-assembler input file specified.
>
> By default, the assembler file name for a source file is made by replacing the suffix .c, .i, etc., with .s.
>
> Input files that don't require compilation are ignored.

> **-fverbose-asm**
>
> Put extra commentary information in the generated assembly code to make it more readable.
> This option is generally only of use to those who actually need to read the generated assembly code
> (perhaps while debugging the compiler itself).
>
> *-fno-verbose-asm*, the default, causes the extra information to be omitted and is useful when comparing two assembler files.

Another option is to compile with debugger options active (`-g` and `-ggdb`) and return backward with `objdump` command:

```shell
gcc -ggdb -std=c99 -Wall -Wpedantic -march=native simd-map-levels.c -o simd-map-levels -I../../../include
objdump -dS simd-map-levels > simd-map-levels.s
```

> **-ggdb**
>
> Produce debugging information for use by GDB.
> This means to use the most expressive format available (DWARF 2, stabs, or the native format if neither of those are supported),
> including GDB extensions if at all possible. 

Flag `-O*` was omitted when compiled in order to avoid code optimizations that would change the code too much.
The same commands could be tested with `-O`, `-O1`, `-O2`, `-O3`, `-Os`... options, in case of specific need.
