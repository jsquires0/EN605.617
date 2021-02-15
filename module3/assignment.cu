#include <stdio.h>
#include <stdlib.h>

/*
	Adds an array containing integers from 0 to totalThreads
	to an array of random integers between [0,3] and stores the 
	result in output array.
*/
__global__
void add(int *pos, int *rnd, int *out)
{
	int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	out[thread_idx] = pos[thread_idx] + rnd[thread_idx];

}
 
    
/*
	Subtracts an array of random integers between [0,3] from
    an array containing integers from 0 to totalThreads
	and stores the result in output array.
*/
__global__
void subtract(int *pos, int *rnd, int *out)
{
	int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	out[thread_idx] = pos[thread_idx] - rnd[thread_idx];

}
   
    
/*
	Performs elementwise multiplication of an array of random 
    integers between [0,3] and an array containing integers 
    from 0 to totalThreads, and stores the result in output array
*/
__global__
void mult(int *pos, int *rnd, int *out)
{
	int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	out[thread_idx] = pos[thread_idx] * rnd[thread_idx];

}

    
/*
	Performs elementwise modular division between 
    an array containing integers from 0 to totalThreads
    and an array of random integers between [0,3].
	Stores the result in output array.
*/
__global__
void mod(int *pos, int *rnd, int *out)
{
	int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	out[thread_idx] = pos[thread_idx] % rnd[thread_idx];

}

extern "C" void doAdd(int numBlocks, int totalThreads, int *pos, int *rnd, int *added)
{
    int *gpu_pos, *gpu_rnd, *gpu_out;
    
    //allocate gpu memory
    cudaMalloc((void**)&gpu_pos, totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_rnd, totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_out, totalThreads * sizeof(int));
    
                                
                                  
    // copy inputs to gpu
    cudaMemcpy(gpu_pos, pos, totalThreads * sizeof(int), 
				cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rnd, rnd, totalThreads * sizeof(int), 
				cudaMemcpyHostToDevice);
    
	add<<<numBlocks, totalThreads>>>(gpu_pos, gpu_rnd, gpu_out);

	// copy back to cpu 
	cudaMemcpy(added, gpu_out, totalThreads * sizeof(int), 
				cudaMemcpyDeviceToHost); 
    
    
	for (int i=0; i<10; i++) 
    {
        printf( "%d + %d = %d\n", pos[i], rnd[i], added[i] );
    }
                                  
	cudaFree(gpu_pos);
	cudaFree(gpu_rnd);
	cudaFree(gpu_out);
    
}