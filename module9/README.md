thrust_math.exe is built with the makefile and run with the command:

thrust_math.exe

The calculation runs twice once using pageable 
host memory, and then again using pinned host memory. This is done
for two sets of input data size.

Timing results are printed to the console for N = 256 and N = 512
