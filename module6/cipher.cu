#include <stdio.h>
#include <stdlib.h>

#define MIN_PRINTABLE 32
#define MAX_PRINTABLE 127
#define PRINTABLE_RANGE (MAX_PRINTABLE - MIN_PRINTABLE + 1)
#define OFFSET 5

#define TOTALTHREADS 1024
#define THREADS_IN_BLOCK 128
__constant__ char const_in_text[TOTALTHREADS];


__host__ cudaEvent_t get_time(void) {
    cudaEvent_t time;
    cudaEventCreate(&time);
    cudaEventRecord(time);
    return time;
}

/**
 * Generates an array of random characters
 */
void fillRandArray(char *input_text, int totalThreads) {
   for (int i = 0; i < TOTALTHREADS; i++)
    {
        int val = rand() % (PRINTABLE_RANGE);              
        input_text[i] = (char) val + MIN_PRINTABLE;                          
    }
}

/**
 * Allocates memory for hosts input and output arrays.
 * Initializes the input array with random characters.                                
 */

 void Alloc(char **input_text, char **result) {
                    
    // allocate host memory
    char *in, *out;
    in = (char*)malloc(TOTALTHREADS*sizeof(char));
    out = (char*)malloc(TOTALTHREADS*sizeof(char));
   
    // populate input array
    fillRandArray(in, TOTALTHREADS);

    // update pointers                           
    *input_text = in;
    *result = out;
}

// ******************************** CONSTANT ******************************* //                                

/**
 * Perform Caesar cipher on an array of characters in parallel.
 * Passing in -OFFSET reverses the operation.
 */
 __global__ void constant_encrypt(char *result) { 
    
    // Calculate the current index
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
     
	/* 
	 * Adjust value of text and key to be based at 0 
	 * Printable ASCII starts at MIN_PRINTABLE, but 0 start is easier to work with 
	 */
    int ascii = const_in_text[idx];
    if (ascii < 32 || ascii > 127)
        printf("Enountered character outside of printable range");
    
	int zeroed_ascii = ascii - MIN_PRINTABLE;
    signed int offset = OFFSET;
    
	// Encrypt by adding the offset value and taking mod to wrap
    int tmp = (zeroed_ascii + offset) % (PRINTABLE_RANGE);
    
    // Handle negative operands..
    int cipherchar = tmp < 0 ? (tmp + PRINTABLE_RANGE) : tmp;
	cipherchar += MIN_PRINTABLE;
	result[idx] = cipherchar;
}

float const_gpu_cipher(int numBlocks, char *input_text, char *result, 
                        char *gpu_out) {

    // copy data from host to gpu
    cudaMemcpyToSymbol(const_in_text, input_text, TOTALTHREADS * sizeof(char));

     // Begin timing
    cudaEvent_t start_time = get_time();
                            
    // compute results on gpu
    constant_encrypt<<<numBlocks, TOTALTHREADS/numBlocks>>>(gpu_out);
    
    // End timing
    cudaEvent_t end_time = get_time();
    cudaEventSynchronize(end_time);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start_time, end_time);
    
    // copy back to cpu 
    cudaMemcpy(result, gpu_out, TOTALTHREADS * sizeof(char), 
    cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);    
                                    
    return elapsed;
}
/**
 * Allocates memory, calls cipher, and cleans up. Input array is stored
 * in constant memory.                             
 */
void const_main(int numBlocks){

    // Initialize input array with random characters
    char *input_text, *result;
    Alloc(&input_text, &result);
    
    // Allocate gpu memory. Don't need malloc gpu_in for constant memory
    char *gpu_out;
    cudaMalloc((void**)&gpu_out, TOTALTHREADS * sizeof(char));

    // Perform encryption
    float elapsed;
    elapsed = const_gpu_cipher(numBlocks,
                    input_text, result, gpu_out);
                        
    // clean up 
    cudaFree(gpu_out);
    free(input_text);
    free(result);

    printf("Constant memory elapsed: %3.3f ms\n", elapsed);                           
}

// ******************************** SHARED ******************************* //

/**
 * Perform Caesar cipher on an array of characters in parallel.
 * Passing in -OFFSET reverses the operation.
 */
 __global__ void shared_encrypt(char *input_text, char *result) { 
    
    __shared__ char shared_in_text[THREADS_IN_BLOCK];

    // Calculate the current index
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int idx_in_block = threadIdx.x;
    shared_in_text[idx_in_block] = input_text[idx];

	/* 
	 * Adjust value of text and key to be based at 0 
	 * Printable ASCII starts at MIN_PRINTABLE, but 0 start is easier to work with 
	 */ 
    int ascii = shared_in_text[idx_in_block];
    if (ascii < 32 || ascii > 127)
        printf("Enountered character outside of printable range");
    
	int zeroed_ascii = ascii - MIN_PRINTABLE;
    signed int offset = OFFSET;
    
	// Encrypt by adding the offset value and taking mod to wrap
    int tmp = (zeroed_ascii + offset) % (PRINTABLE_RANGE);
    
    // Handle negative operands..
    int cipherchar = tmp < 0 ? (tmp + PRINTABLE_RANGE) : tmp;
	cipherchar += MIN_PRINTABLE;
	result[idx] = cipherchar;
}

/**
 * Calls cipher kernel and executes on gpu. Host -> device memory transfer
 * is timed. Shared memory is used for the input array.                  
 */                                 
 float shared_gpu_cipher(int numBlocks, int totalThreads, char *input_text,
    char *result, char *gpu_in, char *gpu_out) {

    // copy data from host to gpu

    cudaMemcpy(gpu_in, input_text, totalThreads * sizeof(char), 
    cudaMemcpyHostToDevice);
    
     // Begin timing
    cudaEvent_t start_time = get_time();
    
    // compute results on gpu
    shared_encrypt<<<numBlocks, totalThreads/numBlocks>>>(gpu_in, gpu_out);
    
     // End timing
    cudaEvent_t end_time = get_time();
    cudaEventSynchronize(end_time);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start_time, end_time);
    
    // copy back to cpu 
    cudaMemcpy(input_text, gpu_in, totalThreads * sizeof(char), 
    cudaMemcpyDeviceToHost);
    cudaMemcpy(result, gpu_out, totalThreads * sizeof(char), 
    cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time); 
                                  
    return elapsed;
} 
/**
 * Allocates pinned memory, calls cipher, and cleans up                             
 */
void shared_main(int numBlocks, int totalThreads) {

    // Initialize input array with random characters
    char *input_text, *result;
    Alloc(&input_text, &result);

    // Allocate gpu memory  
    char *gpu_in, *gpu_out;
    cudaMalloc((void**)&gpu_in, totalThreads * sizeof(char));
    cudaMalloc((void**)&gpu_out, totalThreads * sizeof(char));

    // Perform encryption
    float elapsed;
    elapsed = shared_gpu_cipher(numBlocks, totalThreads,
                input_text, result, gpu_in, gpu_out);
                    
    // clean up                           
    cudaFree(gpu_in);
    cudaFree(gpu_out);
    cudaFree(input_text);
    cudaFree(result);
    
    printf("Shared memory elapsed: %3.3f ms\n", elapsed);                           
}
       
int main(int argc, char** argv) {

	int numBlocks = TOTALTHREADS/THREADS_IN_BLOCK;
    
	// validate command line arguments
	if (TOTALTHREADS % THREADS_IN_BLOCK != 0) {

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("Please update and re-rerun \n");
    }
    
    const_main(numBlocks); 
    shared_main(numBlocks, TOTALTHREADS);

	return EXIT_SUCCESS;
}