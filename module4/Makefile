all: build
build: assignment.exe

kernels.o: kernels.cu
	nvcc -c kernels.cu -L /usr/local/cuda/lib -o kernels.o
main.o: main.cu
	nvcc -c main.cu -L /usr/local/cuda/lib -o main.o
assignment.exe: main.o kernels.o
	nvcc -o assignment.exe main.o kernels.o