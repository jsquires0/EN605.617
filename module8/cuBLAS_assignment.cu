// Computes A*B = C with cuBLAS. A, B, C are NxN matrices
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

int N = 128;

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}
    
/**
 * Allocates pageable memory for host's input and output arrays
 */
 void pageableMathAlloc(int N, float **A, float **B, float **C)
{
// allocate
float *a, *b, *c;
a = (float*)malloc(N*N*sizeof(float));
b = (float*)malloc(N*N*sizeof(float));
c = (float*)malloc(N*N*sizeof(float));

// update pointers                           
*A = a;
*B = b;
*C = c;
}

/**
 * Allocates pinned memory for hosts input and output arrays
 */
 void pinnedMathAlloc(int N, float **A, float **B, float **C)
{
// allocate
float *a, *b, *c;
cudaHostAlloc((void**)&a, N*N*sizeof(float), cudaHostAllocDefault);
cudaHostAlloc((void**)&b, N*N*sizeof(float), cudaHostAllocDefault);           
cudaHostAlloc((void**)&c, N*N*sizeof(float), cudaHostAllocDefault);  

// update pointers                           
*A = a;
*B = b;
*C = c;
}        

void square_matrix_multiplication(int use_pinned, int N)
{
    // allocate pageable host memory, gpu memory
    float *A, *B, *C, *gpu_A, *gpu_B, *gpu_C;
    use_pinned ? pinnedMathAlloc(N, &A, &B, &C) : pageableMathAlloc(N, &A, &B, &C);
    cudaMalloc((void**)&gpu_A, N * N * sizeof(float));
    cudaMalloc((void**)&gpu_B, N * N * sizeof(float));
    cudaMalloc((void**)&gpu_C, N * N * sizeof(float));

    // use cuRAND to populate A, B on gpu
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(rng, gpu_A, N*N);
    curandGenerateUniform(rng, gpu_B, N*N);
   
    cublasHandle_t handle; cublasCreate(&handle);
    // cuBLAS SGEMM computes (k_1 * A) * B + (k_2 * C)
    float k_1 = 1.0f; float k_2 = 0.0f;
    // start timing of kernel + device -> host transfer
    cudaEvent_t start_time = get_time();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &k_1, 
                gpu_A, N, gpu_B, N, &k_2, gpu_C, N);
    
    // copy back to cpu 
	cudaMemcpy(A, gpu_A, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, gpu_B, N*N*sizeof(float), cudaMemcpyDeviceToHost); 
    cudaMemcpy(C, gpu_C, N*N*sizeof(float), cudaMemcpyDeviceToHost); 

    // end timing
    cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	float delta = 0;
    cudaEventElapsedTime(&delta, start_time, end_time);
     use_pinned ? printf("Matmul with pinned mem: %3.3f ms\n", delta) : 
                  printf("Matmul with pageable mem: %3.3f ms\n", delta); 

    // cleanup
    cudaFreeHost(A); cudaFreeHost(B); cudaFreeHost(C);
    cudaFree(gpu_A); cudaFree(gpu_B); cudaFree(gpu_C); cublasDestroy(handle);
    cudaEventDestroy(start_time); cudaEventDestroy(end_time); cudaDeviceReset();
}

/* 
	Computes A*B = C using either pinned or pageable memory for
	two sets matrix sizes
*/
int main(int argc, char** argv) {

	int use_pinned = 1;
    // test one
    square_matrix_multiplication(use_pinned, N);
    square_matrix_multiplication(!use_pinned, N);

	N *= 2;
	// test two
    square_matrix_multiplication(use_pinned, N);
    square_matrix_multiplication(!use_pinned, N);
	
	return EXIT_SUCCESS;
}