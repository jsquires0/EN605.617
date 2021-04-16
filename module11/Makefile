all: build
build: convolution.exe

convolution.exe: Convolution.cpp
	nvcc Convolution.cpp -lOpenCL -o convolution.exe