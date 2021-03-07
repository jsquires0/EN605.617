#include <stdio.h>
#include <stdlib.h>

int TOTALTHREADS = 128;
#define THREADS_IN_BLOCK 128

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

/**
 * Allocates pageable memory for host's input and output arrays
 */
 void MathAlloc( int **pos, int **rnd, 
    int **added, int **subd, 
    int **multd, int **moded)
{
    // allocate
    int *p, *r, *a, *s, *mu, *mo;
    p = (int*)malloc(TOTALTHREADS*sizeof(int));
    r = (int*)malloc(TOTALTHREADS*sizeof(int));
    a = (int*)malloc(TOTALTHREADS*sizeof(int));
    s = (int*)malloc(TOTALTHREADS*sizeof(int));
    mu = (int*)malloc(TOTALTHREADS*sizeof(int));
    mo = (int*)malloc(TOTALTHREADS*sizeof(int));

    // populate input arrays
    for (int i=0; i<TOTALTHREADS; i++)
    {
        p[i] = i;                       
        r[i] = rand() % 4;
    }
    // update pointers                           
    *pos = p;
    *rnd = r;
    *added = a;
    *subd = s;
    *multd = mu;
    *moded = mo;
}

// ******************************** SHARED ******************************* // 
/*
	Adds an array containing integers from 0 to totalThreads
	to an array of random integers between [0,3] and stores the 
	result in output array.
*/
__global__
void sharedAdd(int *pos, int *rnd, int *out)
{
	__shared__ int shared_pos[THREADS_IN_BLOCK];
	__shared__ int shared_rnd[THREADS_IN_BLOCK];

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idx_in_block = threadIdx.x;

	shared_pos[idx_in_block] = pos[idx];
	shared_rnd[idx_in_block] = rnd[idx];

	out[idx] = shared_pos[idx_in_block] + shared_rnd[idx_in_block];
}
 
/*
	Subtracts an array of random integers between [0,3] from
    an array containing integers from 0 to totalThreads
	and stores the result in output array.
*/
__global__
void sharedSubtract(int *pos, int *rnd, int *out)
{
	__shared__ int shared_pos[THREADS_IN_BLOCK];
	__shared__ int shared_rnd[THREADS_IN_BLOCK];

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idx_in_block = threadIdx.x;

	shared_pos[idx_in_block] = pos[idx];
	shared_rnd[idx_in_block] = rnd[idx];

	out[idx] = shared_pos[idx_in_block] - shared_rnd[idx_in_block];
}
     
/*
	Performs elementwise multiplication of an array of random 
    integers between [0,3] and an array containing integers 
    from 0 to totalThreads, and stores the result in output array
*/
__global__
void sharedMult(int *pos, int *rnd, int *out)
{
	__shared__ int shared_pos[THREADS_IN_BLOCK];
	__shared__ int shared_rnd[THREADS_IN_BLOCK];

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idx_in_block = threadIdx.x;

	shared_pos[idx_in_block] = pos[idx];
	shared_rnd[idx_in_block] = rnd[idx];

	out[idx] = shared_pos[idx_in_block] * shared_rnd[idx_in_block];
}
  
/*
	Performs elementwise modular division between 
    an array containing integers from 0 to totalThreads
    and an array of random integers between [0,3].
	Stores the result in output array.
*/
__global__
void sharedMod(int *pos, int *rnd, int *out)
{
	__shared__ int shared_pos[THREADS_IN_BLOCK];
	__shared__ int shared_rnd[THREADS_IN_BLOCK];

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idx_in_block = threadIdx.x;

	shared_pos[idx_in_block] = pos[idx];
	shared_rnd[idx_in_block] = rnd[idx];

	out[idx] = shared_pos[idx_in_block] % shared_rnd[idx_in_block];
}

// ******************************** REGISTER ******************************* //   
/*
	Adds an array containing integers from 0 to totalThreads
	to an array of random integers between [0,3] and stores the 
	result in output array.
*/
__global__
void regAdd(int *pos, int *rnd, int *out)
{
	int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int p_tmp = pos[thread_idx];
	int r_tmp = rnd[thread_idx];
	out[thread_idx] = p_tmp + r_tmp;
}
 
/*
	Subtracts an array of random integers between [0,3] from
    an array containing integers from 0 to totalThreads
	and stores the result in output array.
*/
__global__
void regSubtract(int *pos, int *rnd, int *out)
{
	int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int p_tmp = pos[thread_idx];
	int r_tmp = rnd[thread_idx];
	out[thread_idx] = p_tmp - r_tmp;
}
      
/*
	Performs elementwise multiplication of an array of random 
    integers between [0,3] and an array containing integers 
    from 0 to totalThreads, and stores the result in output array
*/
__global__
void regMult(int *pos, int *rnd, int *out)
{
	int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int p_tmp = pos[thread_idx];
	int r_tmp = rnd[thread_idx];
	out[thread_idx] = p_tmp * r_tmp;
}
   
/*
	Performs elementwise modular division between 
    an array containing integers from 0 to totalThreads
    and an array of random integers between [0,3].
	Stores the result in output array.
*/
__global__
void regMod(int *pos, int *rnd, int *out)
{
	int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int p_tmp = pos[thread_idx];
	int r_tmp = rnd[thread_idx];
	out[thread_idx] = p_tmp % r_tmp;
}
/*
	Execute kernels using register or shared memory
*/
void kernelChoice(int use_reg, int numBlocks, int totalThreads, int *pos,
	int *rnd, int *added, int *subd, int *multd, int *moded)
{
	if (use_reg)
	{
		regAdd<<<numBlocks, totalThreads/numBlocks>>>(pos, rnd, added);
		regSubtract<<<numBlocks, totalThreads/numBlocks>>>(pos, rnd, subd);
		regMult<<<numBlocks, totalThreads/numBlocks>>>(pos, rnd, multd);
		regMod<<<numBlocks, totalThreads/numBlocks>>>(pos, rnd, moded);
	}
	else 
	{
		sharedAdd<<<numBlocks, totalThreads/numBlocks>>>(pos, rnd, added);
		sharedSubtract<<<numBlocks, totalThreads/numBlocks>>>(pos, rnd, subd);
		sharedMult<<<numBlocks, totalThreads/numBlocks>>>(pos, rnd, multd);
		sharedMod<<<numBlocks, totalThreads/numBlocks>>>(pos, rnd, moded);
	}
}

/* 
    Calls add, subtract, mult, and mod and performs calculations on gpu
*/
float doMath(int use_reg, int numBlocks, int totalThreads, int *pos, 
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
	cudaMemcpy(gpu_pos, pos, totalThreads * sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_rnd, rnd, totalThreads * sizeof(int),cudaMemcpyHostToDevice);
    
	cudaEvent_t start_time = get_time();

	// compute results on gpu 
	kernelChoice(use_reg, numBlocks, totalThreads, gpu_pos, gpu_rnd, 
				 gpu_added, gpu_subd, gpu_multd, gpu_moded);

	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);

	// copy back to cpu 
	cudaMemcpy(added,gpu_added,totalThreads*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(subd, gpu_subd,totalThreads*sizeof(int), cudaMemcpyDeviceToHost); 
    cudaMemcpy(multd,gpu_multd,totalThreads*sizeof(int),cudaMemcpyDeviceToHost); 
    cudaMemcpy(moded,gpu_moded,totalThreads*sizeof(int),cudaMemcpyDeviceToHost); 
	         
    // clean up                           
	cudaFree(gpu_pos); cudaFree(gpu_rnd); cudaFree(gpu_added); 
	cudaFree(gpu_subd); cudaFree(gpu_multd); cudaFree(gpu_moded); 
	cudaEventDestroy(start_time); cudaEventDestroy(end_time); cudaDeviceReset();
	
    return delta;
}

/* 
    Wrapper to save output to file and print timing metrics
*/
void sub_main(int use_reg, int numBlocks, int totalThreads)
{
    
    int *pos, *rnd, *added, *subd, *multd, *moded;
    MathAlloc(&pos, &rnd, &added, &subd, &multd, &moded);

    // add, subtract, mult, and mod the two input arrays

    float elapsed;
    elapsed = doMath(use_reg, numBlocks, totalThreads, pos, rnd, added, subd, 
						   multd, moded);

	if (use_reg)
	{
		printf("Register memory elapsed: %3.3f ms\n", elapsed); 
	}
	else
	{
		printf("Shared memory elapsed: %3.3f ms\n", elapsed); 
	}

    // Save results
    FILE * outFile;
    outFile = fopen("computed_arrays.txt","w");
    for (int i=0; i<totalThreads; i++)
    {
        fprintf(outFile, "%d\t %d\t %d\t %d\t %d\t %d\t \n", 
                pos[i], rnd[i], added[i], subd[i], multd[i], moded[i]);
    }
    
}

/* 
	Calls all math kernels using either register or shared memory for
	two sets of thread and block sizes.
*/
int main(int argc, char** argv) {

	int numBlocks = TOTALTHREADS/THREADS_IN_BLOCK;
    
	// validate arguments
	if (TOTALTHREADS % THREADS_IN_BLOCK != 0) {

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("Please update and re-run \n");
	}
	int use_reg = 1;
    // test one
    sub_main(use_reg, numBlocks, TOTALTHREADS);
    sub_main(!use_reg, numBlocks, TOTALTHREADS);

	TOTALTHREADS *= 10;
	numBlocks *= 10;
	
	// test two
	sub_main(use_reg, numBlocks, TOTALTHREADS);
    sub_main(!use_reg, numBlocks, TOTALTHREADS);
	
	return EXIT_SUCCESS;
}