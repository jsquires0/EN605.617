all: build
build: math_assignment.exe cipher_assignment.exe

cipher_assignment.exe: cipher.cu
	nvcc cipher.cu -L /usr/local/cuda/lib -o cipher_assignment.exe

math_assignment.exe: math_main.cu
	nvcc math_main.cu -L /usr/local/cuda/lib -o math_assignment.exe