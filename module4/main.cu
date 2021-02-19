#include <stdio.h>
#include <stdlib.h>

void doMath(int numBlocks, int totalThreads, int *pos, 
            int *rnd, int *added, int *subd, int *multd, 
            int *moded);

__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}
    
/**
 * Allocates pageable memory for host's input and output arrays
 */
void pageableMathAlloc(int totalThreads, int **pos,
                int **rnd, int **added, int **subd, 
                int **multd, int **moded)
{
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
    *pos = p;
    *rnd = r;
    *added = a;
    *subd = s;
    *multd = mu;
    *moded = mo;
}

void main_sub0(int numBlocks, int totalThreads)
{
    
    int *pos, *rnd, *added, *subd, *multd, *moded;
    pageableMathAlloc(totalThreads, &pos, &rnd, &added, &subd, &multd, &moded);
    cudaEvent_t start_time = get_time();
    
    // add, subtract, mult, and mod the two input arrays
    doMath(numBlocks, totalThreads, pos, rnd, added, subd, multd, moded);

    cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);

	printf("elapsed time with pageable mem: %3.1f ms\n", delta);
                                  
    // Save results
    FILE * outFile;
    outFile = fopen("computed_arrays.txt","w");
    for (int i=0; i<totalThreads; i++)
    {
        fprintf(outFile, "%d\t %d\t %d\t %d\t %d\t %d\t \n", 
                pos[i], rnd[i], added[i], subd[i], multd[i], moded[i]);
    }
    
}

int main(int argc, char** argv)
{
    // read command line arguments
	int totalThreads = (1 << 20);
    //int totalThreads = 64;                              
	int blockSize = 32;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;
    
	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
    
    main_sub0(numBlocks, totalThreads);
        
	return EXIT_SUCCESS;
}