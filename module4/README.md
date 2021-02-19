To run:
make
assignment.exe <total threads> <block size>

For example, executing
assignment.exe 512 256
will perform add, sub, mult, and mod on two arrays of size 512 using 512 threads and 2 blocks.

Output is saved in computed_arrays.txt formatted as:
input_arr1, input_arr2, added, subtracted, multiplied, mod'd

Examples can be found in the saved_outputs directory
  64 threads, 32 blocksize <computed_arrays_64_32.txt>
  64 threads, 8 blocksize <computed_arrays_64_8.txt>
  512 threads, 256 blocksize <computed_arrays_512_256.txt>
  1024 threads, 256 blocksize <computed_arrays_1024_256.txt>
