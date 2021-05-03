all: build
build: events.exe

average.exe: events.cpp
	nvcc events.cpp -lOpenCL -o events.exe
