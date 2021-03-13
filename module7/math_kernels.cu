#include <stdio.h>
#include <stdlib.h>

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

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
float doMath(cudaStream_t stream_0, cudaStream_t stream_1,  int totalThreads, int batchSize, int blockSize,
	int *pos, int *rnd, int *added, int *subd, int *multd, int *moded)
{
	int *dev_0_pos, *dev_1_pos, *dev_0_rnd, *dev_1_rnd;
	int	*dev_0_added, *dev_1_added, *dev_0_subd, *dev_1_subd;
	int	*dev_0_multd, *dev_1_multd, *dev_0_moded, *dev_1_moded;
    
    //allocate gpu memory
    cudaMalloc((void**)&dev_0_pos, batchSize * sizeof(int));
    cudaMalloc((void**)&dev_0_rnd, batchSize * sizeof(int));
    cudaMalloc((void**)&dev_0_added, batchSize * sizeof(int));
    cudaMalloc((void**)&dev_0_subd, batchSize * sizeof(int));
    cudaMalloc((void**)&dev_0_multd, batchSize * sizeof(int));
    cudaMalloc((void**)&dev_0_moded, batchSize * sizeof(int));
	
	cudaMalloc((void**)&dev_1_pos, batchSize * sizeof(int));
    cudaMalloc((void**)&dev_1_rnd, batchSize * sizeof(int));
    cudaMalloc((void**)&dev_1_added, batchSize * sizeof(int));
    cudaMalloc((void**)&dev_1_subd, batchSize * sizeof(int));
    cudaMalloc((void**)&dev_1_multd, batchSize * sizeof(int));
	cudaMalloc((void**)&dev_1_moded, batchSize * sizeof(int));
	
    cudaEvent_t start_time = get_time();
	// process totalThreads into batches of 16
	for (int i = 0; i<totalThreads, i+= 2*batchSize) {
	
		// copy inputs to GPU
		cudaMemcpyAsync(dev_0_pos, pos_i+i, batchSize * sizeof(int), 
					cudaMemcpyHostToDevice, stream_0);
		cudaMemcpyAsync(dev_1_pos, pos_i+i+batchSize, batchSize * sizeof(int), 
					cudaMemcpyHostToDevice, stream_1);		
		cudaMemcpyAsync(dev_0_rnd, rnd_i+i, batchSize * sizeof(int), 
					cudaMemcpyHostToDevice, stream_0);
		cudaMemcpyAsync(dev_1_rnd, rnd_i+i+batchSize, batchSize * sizeof(int), 
					cudaMemcpyHostToDevice, stream_1);	

		int numBlocks = batchSize/blockSize;
    	// compute batch results on gpu
		add<<<numBlocks, blockSize, 0, stream_0>>>(dev_0_pos, dev_0_rnd, dev_0_added);
		add<<<numBlocks, blockSize, 0, stream_1>>>(dev_1_pos, dev_1_rnd, dev_1_added);
		subtract<<<numBlocks, blockSize, 0, stream_0>>>(dev_0_pos, dev_0_rnd, dev_0_subd);
		subtract<<<numBlocks, blockSize, 0, stream_1>>>(dev_1_pos, dev_1_rnd, dev_1_subd);
		mult<<<numBlocks, blockSize, 0, stream_0>>>(dev_0_pos, dev_0_rnd, dev_0_multd);
		mult<<<numBlocks, blockSize, 0, stream_1>>>(dev_1_pos, dev_1_rnd, dev_1_multd);
		mod<<<numBlocks, blockSize, 0, stream_0>>>(dev_0_pos, dev_0_rnd, dev_0_moded);
		mod<<<numBlocks, blockSize, 0, stream_1>>>(dev_1_pos, dev_1_rnd, dev_1_moded);
		
		// copy batch back to cpu 
		cudaMemcpyAsync(added+i,dev_0_added,batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_0);
		cudaMemcpyAsync(added+i+batchSize,dev_1_added,batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_1);
		cudaMemcpyAsync(subd, dev_0_subd,batchSize*sizeof(int), cudaMemcpyDeviceToHost, stream_0);
		cudaMemcpyAsync(subd+i+batchSize,dev_1_subd,batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_1);
		cudaMemcpyAsync(multd,dev_0_multd,batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_0);
		cudaMemcpyAsync(multd+i+batchSize,dev_1_multd,batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_1);
		cudaMemcpyAsync(moded,dev_0_moded,batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_0);
		cudaMemcpyAsync(moded+i+batchSize,dev_1_moded,batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_1);
	}
	cudaStreamSynchronize(stream_0); cudaStreamSynchronize(stream_1);
	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);

    // clean up                           
	cudaFree(dev_0_pos); cudaFree(dev_0_rnd); cudaFree(dev_0_added); 
	cudaFree(dev_0_subd); cudaFree(dev_0_multd); cudaFree(dev_0_moded);
	cudaFree(dev_1_pos); cudaFree(dev_1_rnd); cudaFree(dev_1_added); 
	cudaFree(dev_1_subd); cudaFree(dev_1_multd); cudaFree(dev_1_moded);  
	cudaEventDestroy(start_time); cudaEventDestroy(end_time); cudaDeviceReset();
    return delta;
}