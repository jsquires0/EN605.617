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
	Allocates GPU memory for both streams 
*/
void multi_stream_malloc(int batchSize, 
						 int **dev_0_pos, int **dev_1_pos, 
						 int **dev_0_rnd, int **dev_1_rnd,
						 int **dev_0_added, int **dev_1_added, 
						 int **dev_0_subd, int **dev_1_subd,
						 int **dev_0_multd, int **dev_1_multd,
						 int **dev_0_moded, int **dev_1_moded)
{
	//allocate
	int *d0p, *d1p, *d0r, *d1r, *d0a, *d1a;
	int *d0s, *d1s, *d0mu, *d1mu, *d0mo, *d1mo;
    cudaMalloc((void**)&d0p, batchSize * sizeof(int));
    cudaMalloc((void**)&d0r, batchSize * sizeof(int));
    cudaMalloc((void**)&d0a, batchSize * sizeof(int));
    cudaMalloc((void**)&d0s, batchSize * sizeof(int));
    cudaMalloc((void**)&d0mu, batchSize * sizeof(int));
    cudaMalloc((void**)&d0mo, batchSize * sizeof(int));
	cudaMalloc((void**)&d1p, batchSize * sizeof(int));
    cudaMalloc((void**)&d1r, batchSize * sizeof(int));
    cudaMalloc((void**)&d1a, batchSize * sizeof(int));
    cudaMalloc((void**)&d1s, batchSize * sizeof(int));
    cudaMalloc((void**)&d1mu, batchSize * sizeof(int));
    cudaMalloc((void**)&d1mo, batchSize * sizeof(int));

	// update pointers
	*dev_0_pos = d0p; *dev_1_pos = d1p;
	*dev_0_rnd = d0r; *dev_1_rnd = d1r;
	*dev_0_added = d0a; *dev_1_added = d1a;
	*dev_0_subd = d0s; *dev_1_subd = d1s;
	*dev_0_multd = d0mu; *dev_1_multd = d1mu;
	*dev_0_moded = d0mo; *dev_1_moded = d0mo;
}
/* 
	Copy host input data to cuda device streams
*/
void  async_host_to_dev(cudaStream_t stream_0, cudaStream_t stream_1, 
	int iteration, int batchSize, int *pos, int *rnd, int *dev_0_pos, 
	int *dev_1_pos, int *dev_0_rnd, int *dev_1_rnd)
{
	cudaMemcpyAsync(dev_0_pos, pos+iteration, 
					batchSize * sizeof(int), cudaMemcpyHostToDevice, stream_0);
	cudaMemcpyAsync(dev_1_pos, pos+iteration+batchSize, 
					batchSize * sizeof(int), cudaMemcpyHostToDevice, stream_1);		
	cudaMemcpyAsync(dev_0_rnd, rnd+iteration, 
					batchSize * sizeof(int), cudaMemcpyHostToDevice, stream_0);
	cudaMemcpyAsync(dev_1_rnd, rnd+iteration+batchSize, 
					batchSize * sizeof(int), cudaMemcpyHostToDevice, stream_1);	
}
/* 
    Performs add, subtract, mult, and mod calculations on gpu
*/
void call_kernels(cudaStream_t stream_0, cudaStream_t stream_1, 
				  int batchSize, int blockSize, 
				  int *dev_0_pos, int *dev_1_pos, 
				  int *dev_0_rnd, int *dev_1_rnd,
				  int *dev_0_added, int *dev_1_added, 
				  int *dev_0_subd, int *dev_1_subd,
				  int *dev_0_multd, int *dev_1_multd,
				  int *dev_0_moded, int *dev_1_moded)
{
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
}

/* 
	Copies GPU results back to host
 */
void async_dev_to_host(cudaStream_t stream_0, cudaStream_t stream_1, 
	int iteration, int batchSize, int *added, int *subd, int *multd, int *moded, 
	int *dev_0_added, int *dev_1_added, int *dev_0_subd, int *dev_1_subd, 
	int *dev_0_multd, int *dev_1_multd, int *dev_0_moded, int *dev_1_moded)
{
	cudaMemcpyAsync(added+iteration, dev_0_added, 
					batchSize*sizeof(int), cudaMemcpyDeviceToHost, stream_0);
	cudaMemcpyAsync(added+iteration+batchSize, dev_1_added, 
					batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_1);
	cudaMemcpyAsync(subd+iteration, dev_0_subd,
					batchSize*sizeof(int), cudaMemcpyDeviceToHost, stream_0);
	cudaMemcpyAsync(subd+iteration+batchSize, dev_1_subd,
					batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_1);
	cudaMemcpyAsync(multd+iteration, dev_0_multd,
					batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_0);
	cudaMemcpyAsync(multd+iteration+batchSize, dev_1_multd,
					batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_1);
	cudaMemcpyAsync(moded+iteration, dev_0_moded,
					batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_0);
	cudaMemcpyAsync(moded+iteration+batchSize, dev_1_moded,
					batchSize*sizeof(int),cudaMemcpyDeviceToHost, stream_1);
}
		  
/* 
	Allocates gpu memory, copies host -> device, performs math calculations,
	copies device -> host, cleans up
*/
float doMath(cudaStream_t stream_0, cudaStream_t stream_1,  int totalThreads, 
			 int batchSize, int blockSize, int *pos, int *rnd, int *added, 
			 int *subd, int *multd, int *moded)
{
	// allocate gpu memory for each stream
	int *dev_0_pos, *dev_1_pos, *dev_0_rnd, *dev_1_rnd;
	int	*dev_0_added, *dev_1_added, *dev_0_subd, *dev_1_subd;
	int	*dev_0_multd, *dev_1_multd, *dev_0_moded, *dev_1_moded;
	multi_stream_malloc(batchSize, &dev_0_pos, &dev_1_pos, &dev_0_rnd, 
		&dev_1_rnd, &dev_0_added, &dev_1_added, &dev_0_subd, &dev_1_subd, 
		&dev_0_multd, &dev_1_multd, &dev_0_moded, &dev_1_moded);
    
    cudaEvent_t start_time = get_time();
	// process totalThreads in batches
	for (int i = 0; i<totalThreads; i+= 2*batchSize) {
		// copy batch to GPU
		async_host_to_dev(stream_0, stream_1, i, batchSize, pos, rnd, dev_0_pos,
			dev_1_pos, dev_0_rnd, dev_1_pos);
        // do calculations on batch                     
		call_kernels(stream_0, stream_1, batchSize, blockSize, dev_0_pos, dev_1_pos, 
			dev_0_rnd, dev_1_rnd, dev_0_added, dev_1_added, dev_0_subd, dev_1_subd,
			dev_0_multd, dev_1_multd, dev_0_moded, dev_1_moded);
		// copy batch back to cpu 
		async_dev_to_host(stream_0, stream_1, i, batchSize, added, subd, multd, 
			moded, dev_0_added, dev_1_added, dev_0_subd, dev_1_subd, 
			dev_0_multd, dev_1_multd, dev_0_moded, dev_1_moded);
	}
    // synchronize streams, get elapsed time
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