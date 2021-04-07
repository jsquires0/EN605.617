all: build
build: openCL_math.exe

openCL_math.exe: openCL_math.cpp
	nvcc openCL_math.cpp -lOpenCL -o openCL_math.exe