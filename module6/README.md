The submission is divided into two separate targets:
cipher_assignment.exe and math_assignment.exe
Both are built with the same Makefile, and run with the same commands:

*_assignment.exe

For example, executing

cipher_assignment.exe 

will encrypt a random input array of TOTALTHREADS characters using the cipher. 
The calculation runs on TOTALTHREADS threads and TOTALTHREADS/THREADS_IN_BLOCK 
blocks once using constant memory, and then again using shared memory. The main
function then runs the kernels again, with 10 times the specified number
of threads and blocks. 

The values of TOTALTHREADS and THREADS_IN_BLOCK can be found at the top of
cipher.cu and math_main.cu. TOTALTHREADS must be divisible by THREADS_IN_BLOCK.


Kernel execution times are printed to the console.
