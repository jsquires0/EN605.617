#include <stdio.h>
#include <stdlib.h>

int TOTAL_THREADS = 8192;
int BLOCK_SIZE = 128;
int BATCH_SIZE = 1024;

float doMath(cudaStream_t stream_0, cudaStream_t stream_1, int totalThreads, 
            int batchSize, int blockSize, int *pos, 
            int *rnd, int *added, int *subd, int *multd, 
            int *moded);
    
/**
 * Allocates pageable memory for host's input and output arrays
 */
void pageableMathAlloc(int totalThreads, int **pos,
                int **rnd, int **added, int **subd, 
                int **multd, int **moded)
{
    // allocate
    int *p, *r, *a, *s, *mu, *mo;
    p = (int*)malloc(totalThreads*sizeof(int));
    r = (int*)malloc(totalThreads*sizeof(int));
    a = (int*)malloc(totalThreads*sizeof(int));
    s = (int*)malloc(totalThreads*sizeof(int));
    mu = (int*)malloc(totalThreads*sizeof(int));
    mo = (int*)malloc(totalThreads*sizeof(int));

    // populate input arrays
    for (int i=0; i<totalThreads; i++)
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
                                  
/**
 * Allocates pinned memory for hosts input and output arrays
 */
void pinnedMathAlloc(int totalThreads, int **pos, int **rnd, int **added, 
                     int **subd, int **multd, int **moded)
{
    // allocate
    int *p, *r, *a, *s, *mu, *mo;
    cudaHostAlloc((void**)&p,totalThreads*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&r,totalThreads*sizeof(int), cudaHostAllocDefault);           
    cudaHostAlloc((void**)&a,totalThreads*sizeof(int), cudaHostAllocDefault);  
    cudaHostAlloc((void**)&s,totalThreads*sizeof(int), cudaHostAllocDefault); 
    cudaHostAlloc((void**)&mu,totalThreads*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&mo,totalThreads*sizeof(int), cudaHostAllocDefault);

    // populate input arrays
    for (int i=0; i<totalThreads; i++)
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

       
void pageable_sub(int totalThreads, int blockSize, int batchSize)
{
    // allocate pageable host memory
    int *pos, *rnd, *added, *subd, *multd, *moded;
    pageableMathAlloc(totalThreads, &pos, &rnd, &added, &subd, &multd, &moded);
    
    // initialize streams
    cudaDeviceProp prop;
    int whichDevice; 

    cudaGetDeviceCount( &whichDevice); 
    cudaGetDeviceProperties( &prop, whichDevice); 

    cudaStream_t stream_0, stream_1; 
    cudaStreamCreate(&stream_0); cudaStreamCreate(&stream_1);
    
    // add, subtract, mult, and mod the two input arrays
    // Time host -> device, kernel execution, device -> host
    float elapsed;
    elapsed = doMath(stream_0, stream_1, totalThreads, batchSize, blockSize, 
                     pos, rnd, added, subd, multd, moded);
    printf("Host -> device transfer with pageable mem: %3.3f ms\n", elapsed);    

    // Save results
    FILE * outFile;
    outFile = fopen("computed_arrays.txt","w");
    for (int i=0; i<totalThreads; i++)
    {
        fprintf(outFile, "%d\t %d\t %d\t %d\t %d\t %d\t \n", 
                pos[i], rnd[i], added[i], subd[i], multd[i], moded[i]);
    }
                                  
    cudaFreeHost(pos); 
    cudaFreeHost(rnd);
    
}

void pinned_sub(int totalThreads, int blockSize, int batchSize)
{
    // allocate host memory
    int *pos, *rnd, *added, *subd, *multd, *moded;
    pinnedMathAlloc(totalThreads, &pos, &rnd, &added, &subd, &multd, &moded);
   
    // initialize streams
    cudaDeviceProp prop;
    int whichDevice; 

    cudaGetDeviceCount( &whichDevice); 
    cudaGetDeviceProperties( &prop, whichDevice); 

    cudaStream_t stream_0, stream_1; 
    cudaStreamCreate(&stream_0); cudaStreamCreate(&stream_1);

    // add, subtract, mult, and mod the two input arrays
    // Time host -> device, kernel execution, device -> host
    float elapsed;
    elapsed = doMath(stream_0, stream_1, totalThreads, batchSize, blockSize, 
                     pos, rnd, added, subd, multd, moded);

    printf("Host -> device transfer with pinned mem: %3.3f ms\n", elapsed);
    cudaFreeHost(pos); 
    cudaFreeHost(rnd);
    
}                                  

/* 
    Calls all math kernels using either pinned or pageable host memory,
    two sets of thread and block sizes. Uses two cuda streams so data may
    be copied to gpu as kernel is processed.
*/
int main(int argc, char** argv)
{
    
	// validate command line arguments
	if (TOTAL_THREADS % BLOCK_SIZE != 0) {

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("Please update and re-run \n");
	}
    
    pageable_sub(TOTAL_THREADS, BLOCK_SIZE, BATCH_SIZE); 
    pinned_sub(TOTAL_THREADS, BLOCK_SIZE, BATCH_SIZE); 
    
    TOTAL_THREADS *= 2;
    BLOCK_SIZE *= 2;

    pageable_sub(TOTAL_THREADS, BLOCK_SIZE, BATCH_SIZE); 
    pinned_sub(TOTAL_THREADS, BLOCK_SIZE, BATCH_SIZE); 

	return EXIT_SUCCESS;
}