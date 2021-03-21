math_assignment.exe is built with the makefile and run with the command:

math_assignment.exe

The calculation runs on 4096 threads and 16 blocks twice: once using pageable 
host memory, and then again using pinned host memory. This is done
for two sets of total threads and blocksize combinations. Data is processed
on two cuda streams, using batch sizes of 1024.

Timing results are printed to the console for:

TOTAL_THREADS = 8192, BLOCK_SIZE = 128

and

TOTAL_THREADS = 16384, BLOCK_SIZE = 256

Note totalthreads MUST be a multiple of the batch size, 
and blocksize should be a divisor of batch size.

Since using multiple cuda streams in parallel requires pinned host memory, 
the pageable implementation is upwards of 2x slower.
