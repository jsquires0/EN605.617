#include <stdio.h>
#include <stdlib.h>

float doMath(int numBlocks, int totalThreads, int *pos, 
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
void pinnedMathAlloc(int totalThreads, int **pos,
                int **rnd, int **added, int **subd, 
                int **multd, int **moded)
{
    // allocate
    int *p, *r, *a, *s, *mu, *mo;
    cudaHostAlloc((void**)&p,
                       totalThreads*sizeof(int),
                       cudaHostAllocDefault);
    cudaHostAlloc((void**)&r,
                       totalThreads*sizeof(int),
                       cudaHostAllocDefault);                             
    cudaHostAlloc((void**)&a,
                       totalThreads*sizeof(int),
                       cudaHostAllocDefault);   
    cudaHostAlloc((void**)&s,
                       totalThreads*sizeof(int),
                       cudaHostAllocDefault);   
    cudaHostAlloc((void**)&mu,
                       totalThreads*sizeof(int),
                       cudaHostAllocDefault);   
    cudaHostAlloc((void**)&mo,
                       totalThreads*sizeof(int),
                       cudaHostAllocDefault);

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

    
    
    
void pageable_sub(int numBlocks, int totalThreads)
{
    
    int *pos, *rnd, *added, *subd, *multd, *moded;
    pageableMathAlloc(totalThreads, &pos, &rnd, &added, &subd, &multd, &moded);

    // add, subtract, mult, and mod the two input arrays

    float elapsed;
    elapsed = doMath(numBlocks, totalThreads, pos, rnd, added, subd, multd, moded);
    printf("Host -> device transfer with pageable mem: %3.3f ms\n", elapsed);                           
    // Save results
    FILE * outFile;
    outFile = fopen("computed_arrays.txt","w");
    for (int i=0; i<totalThreads; i++)
    {
        fprintf(outFile, "%d\t %d\t %d\t %d\t %d\t %d\t \n", 
                pos[i], rnd[i], added[i], subd[i], multd[i], moded[i]);
    }
    
}

void pinned_sub(int numBlocks, int totalThreads)
{
    
    int *pos, *rnd, *added, *subd, *multd, *moded;
    pinnedMathAlloc(totalThreads, &pos, &rnd, &added, &subd, &multd, &moded);
    
    // add, subtract, mult, and mod the two input arrays
    // Time up copy
    float elapsed;
    elapsed = doMath(numBlocks, totalThreads, pos, rnd, added, subd, multd, moded);
    printf("Host -> device transfer with pinned mem: %3.3f ms\n", elapsed);  

                                  
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
    
    pageable_sub(numBlocks, totalThreads);
    pinned_sub(numBlocks, totalThreads);    
	return EXIT_SUCCESS;
}