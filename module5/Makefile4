all: build
build: math_assignment.exe cipher_assignment.exe

cipher_assignment.exe: cipher.cu
	nvcc cipher.cu -L /usr/local/cuda/lib -o cipher_assignment.exe

math_kernels.o: math_kernels.cu
	nvcc -c math_kernels.cu -L /usr/local/cuda/lib -o math_kernels.o
math_main.o: math_main.cu
	nvcc -c math_main.cu -L /usr/local/cuda/lib -o math_main.o
math_assignment.exe: math_main.o math_kernels.o
	nvcc -o math_assignment.exe math_main.o math_kernels.o