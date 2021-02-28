The submission is divided into two separate targets:
cipher_assignment.exe and math_assignment.exe
Both are built with the same Makefile, and run with the same commands:

*_assignment.exe <total threads> <block size>

For example, executing

cipher_assignment.exe 512 256

will encrypt a random input array of 512 characters using the cipher. The calculation runs on 512 threads and 2 blocks twice: once using pageable host memory, and then again using pinned host memory.

Timing results are printed to the console. The 
host -> device transfer is timed, though there's no particular reason to choose this over device -> host. Both show
the same result: pinned memory is around 1.5-2x faster for large transfers
