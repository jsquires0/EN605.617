all: build
build: cuSPARSE_assignment.exe cuBLAS_assignment.exe

cuSPARSE_assignment.exe: cuSPARSE_assignment.cu
	nvcc cuSPARSE_assignment.cu -lcudart -lcuda -lcusparse -I common/inc -o cuSPARSE_assignment.exe

cuBLAS_assignment.exe: cuBLAS_assignment.cu
	nvcc cuBLAS_assignment.cu -lcudart -lcuda -lcublas -lcurand -I common/inc -o cuBLAS_assignment.exe