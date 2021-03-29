all: build
build: thrust_math.exe

thrust_math.exe: thrust_math.cu
	nvcc thrust_math.cu -ccbin g++ -o thrust_math.exe
