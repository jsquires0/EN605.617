//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>


void doAdd(int numBlocks, int totalThreads, int *pos, int *rnd, int *added);

int main(int argc, char** argv)
{
    // read command line arguments
	//int totalThreads = (1 << 20);
    int totalThreads = 64;                              
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
    
    
	
    int pos[totalThreads], rnd[totalThreads], added[totalThreads];
    int subd[totalThreads], multed[totalThreads], moded[totalThreads];
    
    // populate input arrays
	for (int i=0; i<totalThreads; i++)
	{
		pos[i] = i;                       
		rnd[i] = rand() % 4;
	}
    
    doAdd(numBlocks, totalThreads, pos, rnd, added);

	return EXIT_SUCCESS;
}