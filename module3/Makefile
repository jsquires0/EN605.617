all: build
build: assignment.exe

assignment.o: assignment.cu
	gcc -c assignment.c -o assignment.o
cuda.o: assignment.cu
	nvcc -c assignment.cu -L /usr/local/cuda/lib -o cuda.o
assignment.exe: assignment.o cuda.o
	nvcc -o assignment.exe assignment.o cuda.o