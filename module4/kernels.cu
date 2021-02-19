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

    
/* 
    Calls add, subtract, mult, and mod and performs calculations on gpu
*/
extern "C" void doMathPageable(int numBlocks, int totalThreads, int *pos, 
                       int *rnd, int *added, int *subd, int *multd, int *moded)
{
    int *gpu_pos, *gpu_rnd, *gpu_added, *gpu_subd, *gpu_multd, *gpu_moded;
    
    //allocate gpu memory
    cudaMalloc((void**)&gpu_pos, totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_rnd, totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_added, totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_subd, totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_multd, totalThreads * sizeof(int));
    cudaMalloc((void**)&gpu_moded, totalThreads * sizeof(int));
                  
    // copy inputs to gpu
    cudaMemcpy(gpu_pos, pos, totalThreads * sizeof(int), 
				cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rnd, rnd, totalThreads * sizeof(int), 
				cudaMemcpyHostToDevice);
    
    // compute results on gpu
	add<<<numBlocks, totalThreads>>>(gpu_pos, gpu_rnd, gpu_added);
    subtract<<<numBlocks, totalThreads>>>(gpu_pos, gpu_rnd, gpu_subd);
    mult<<<numBlocks, totalThreads>>>(gpu_pos, gpu_rnd, gpu_multd);
    mod<<<numBlocks, totalThreads>>>(gpu_pos, gpu_rnd, gpu_moded);
    
	// copy back to cpu 
	cudaMemcpy(added, gpu_added, totalThreads * sizeof(int), 
				cudaMemcpyDeviceToHost);
    cudaMemcpy(subd, gpu_subd, totalThreads * sizeof(int), 
				cudaMemcpyDeviceToHost); 
    cudaMemcpy(multd, gpu_multd, totalThreads * sizeof(int), 
				cudaMemcpyDeviceToHost); 
    cudaMemcpy(moded, gpu_moded, totalThreads * sizeof(int), 
				cudaMemcpyDeviceToHost); 
                        
    // clean up                           
	cudaFree(gpu_pos);
	cudaFree(gpu_rnd);
	cudaFree(gpu_added);
    cudaFree(gpu_subd);
    cudaFree(gpu_multd);
    cudaFree(gpu_moded);
    
}