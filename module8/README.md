The submission is divided into two separate targets:
cuBLAS_assignment.cu and cuSPARSE_assignment.cu
Both are built with the same Makefile, and run with the same commands:

*_assignment.exe

cuBLAS_assignment.cu 
computes the matrix multiplication A*B = C using cublas for random matrices A, B.

cuSPARSE_assignment.cu
Takes a half filled dense matrix A and converts it into CSR sparse format
with cusparse.

For both .exe's, kernel execution times are printed to the console for two
input sizes for both pageable and pinned host memory

NOTE:: I wasn't able to get cuSPARSE_assignment.cu to compile on vocareum. Some 
of the cusparse functions used are only supported by more recent versions
of cuda. I successfully compiled and ran cuSPARSE_assignment.cu on a cuda 11.2
system.