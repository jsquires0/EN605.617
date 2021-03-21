// Converts a dense NxN matrix A into CSR format using cuSPARSE
#include <stdio.h>
#include <stdlib.h>
#include <cusparse.h>

int N = 128;

/**
 * Allocates pageable memory for host's input and output arrays
 */
 void pageableNAlloc(int N, float **A, int **row)
{
// allocate
float *a; int *r;
a = (float*)malloc(N*N*sizeof(float));
o = (float*)malloc((N+1)*sizeof(float));

// populate input array
for (int i=0; i< N*N; i++)
{  
    // 0.5 density
    if (i % 2 == 0){
        a[i] = 1.0f; 
    }
    else {
        a[i] = 0.0f;
    }                      
}

// update pointers                           
*A = a;
*row = r;
}
/**
 * Allocates pageable memory for host's input and output arrays
 */
 void pageableNNZAlloc(int nnz, float **val, int **col)
{
// allocate
float *v; int *c;
v = (float*)malloc(nnz*sizeof(float));
c = (float*)malloc(nnz*sizeof(float));

// update pointers                           
*col = c;
*val = v;
}

/**
 * Allocates pinned memory for hosts input and output arrays
 */
 void pinnedNAlloc(int N, float **A, int **row)
{
// allocate
float *a; int *r;
cudaHostAlloc((void**)&a, N*N*sizeof(float), cudaHostAllocDefault); 
cudaHostAlloc((void**)&r, (N+1)*sizeof(float), cudaHostAllocDefault);

// populate input array
for (int i=0; i< N*N; i++)
{  
    // 0.5 density
    if (i % 2 == 0){
        a[i] = 1.0f; 
    }
    else {
        a[i] = 0.0f;
    }                      
}
// update pointers                           
*A = a;
*row = r;
}
/**
 * Allocates pageable memory for host's input and output arrays
 */
 void pinnedNNZAlloc(int nnz, float **val, int **col)
{
// allocate
float *v; int *c;
cudaHostAlloc((void**)&v, nnz*sizeof(float), cudaHostAllocDefault); 
cudaHostAlloc((void**)&c, nnz*sizeof(float), cudaHostAllocDefault);

// update pointers                           
*col = c;
*val = v;
}
void dense_to_csr(int use_pinned, int N)
{
    // allocate pageable host memory, gpu memory
    float *DnA, *gpu_DnA, *gpu_val;
    int *row, *gpu_row, *gpu_col, 
    use_pinned ? pinnedNAlloc(N, &DnA, &row) : pageableNAlloc(N, &DnA, &row);
    cudaMalloc((void**)&gpu_DnA, N * N * sizeof(float));
    cudaMalloc((void**)&gpu_row, (N+1) * sizeof(int));

    // copy A, B host -> device
    cudaMemcpy(gpu_DnA, DnA, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    cusparseHandle_t = handle; cusparseCreate(&handle);
    // start timing of kernels + device -> host transfer
    cudaEvent_t start_time = get_time();

    // create matrices
    cusparseDnMatDescr_t dense_mat;
    // initialize dense matrix descriptor
    // signature: https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-dnmat-create
    cusparseCreateDnMat(&dense_mat, N, N, N, gpu_DnA, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    cusparseSpMatDescr_t sparse_mat;
    // initialize sparse matrix descriptor for CSR.
    // signature: https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-spmat-create-csr
    cusparseCreateCsr(&sparse_mat, N, N, 0, gpu_row, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Three steps to convert gpu_DnA -> gpu_SpA (csr format)
    // First, determine size of workspace buffer:
    size_t memBuff = 0, *gpu_memBuff; //WRONG?
    cusparseDenseToSparse_bufferSize(handle, dense_mat, sparse_mat, 
            CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &memBuff);
    cudaMalloc((void**)&gpu_memBuff, memBuff); // WRONG?

    // Second, determine the number of non zero (nnz) elements in A.
    // This is necessary because we haven't yet allocated device memory for gpu_col and gpu_val
    // because their size is equal to the number of non zero (nnz) elements of A.
    // Though in this case we know nnz = N/2, to keep the code flexible we'll get it here:
    cusparseDenseToSparse_analysis(handle, dense_mat, sparse_mat,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, gpu_memBuff);
    
    int64_t* tmp_1, tmp_2, nnz;
    cusparseSpMatGetSize(sparse_mat, &tmp_1, &tmp_2, &nnz);
    cudaMalloc((void**)&gpu_val, nnz * sizeof(float));
    cudaMalloc((void**)&gpu_col, nnz * sizeof(int));

    // Finally, perform the dense -> csr conversion
    // and set gpu_val, gpu_col, gpu_row to point to the result,
    cusparseCsrSetPointers(sparse_mat, gpu_row, gpu_col, gpu_val);
    cusparseDenseToSparse_convert(handle, dense_mat, sparse_mat,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, gpu_memBuff);

    // copy back to cpu
    int *col, float *val;
    use_pinned ? pinnedNNZAlloc(nnz, &val, &col) : pageableNAlloc(nnz, &val, &col);
	cudaMemcpy(row, gpu_row, (N+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(col, gpu_col, nnz * sizeof(int), cudaMemcpyDeviceToHost); 
    cudaMemcpy(val, gpu_val, nnz * sizeof(float), cudaMemcpyDeviceToHost); 

    // end timing
    cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	float delta = 0;
    cudaEventElapsedTime(&delta, start_time, end_time);
    printf("Dense -> Sparse (CSR) with pageable mem: %3.3f ms\n", delta);  

    // cleanup
    cudaaFreeHost(DnA); cudaFreeHost(row); cudaFreeHost(col); cudaFreeHost(val);
    cudaFree(gpu_DnA); cudaFree(gpu_row); cudaFree(gpu_col); cudaFree(gpu_val);
    cudaEventDestroy(start_time); cudaEventDestroy(end_time); cudaDeviceReset();
    cusparseDestroy(handle); cusparseDestroyDnMat(dense_mat); cusparseDestroySpMat(sparse_mat);
}

/* 
    Converts a half empty square matrix into sparse CSR format using either 
    pinned or pageable memory for two sets of matrix sizes
*/
int main(int argc, char** argv) {

	int use_pinned = 1;
    // test one
    dense_to_csr(use_pinned, N);
    dense_to_csr(!use_pinned, N);

	N *= 2;
	// test two
    dense_to_csr(use_pinned, N);
    dense_to_csr(!use_pinned, N);
	
	return EXIT_SUCCESS;
}