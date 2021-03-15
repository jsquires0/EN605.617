math_assignment.exe is built with the makefile and run with the command:

math_assignment.exe

The calculation runs on 4096 threads and 16 blocks twice: once using pageable 
host memory, and then again using pinned host memory. This is done
for two sets of total threads and blocksize combinations.

Timing results are printed to the console for:
TOTAL_THREADS = 8192
BLOCK_SIZE = 128
BATCH_SIZE = 1024

and

TOTAL_THREADS = 16384
BLOCK_SIZE = 256
BATCH_SIZE = 1024

Totalthreads MUST be a multiple of 1024, blocksize should be a divisor of 1024.
