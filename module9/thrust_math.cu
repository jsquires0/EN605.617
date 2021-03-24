#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <iostream>

int N = 256;
typedef thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int> > pin_h_vec;

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

int rand_mod4(void){
    return rand() % 4;
}

/* 
    Host (pinned) -> device, computes math output vectors, 
    device -> host, prints timing metric.
*/
void pinned_sub(int N)
{
    // generate host vecs, populate A with rand, B with sequence
    pin_h_vec h_vecA(N), h_vecB(N);
    thrust::sequence(h_vecA.begin(), h_vecA.end());
    thrust::generate(h_vecB.begin(), h_vecB.end(), rand_mod4);

    // initialize output host vectors
    pin_h_vec h_vecAdd(N), h_vecSub(N);
    pin_h_vec h_vecMult(N), h_vecMod(N);

    // transfer inputs to device, init outputs
    thrust::device_vector<int> d_vecA = h_vecA, d_vecB = h_vecB;
    thrust::device_vector<int> d_vecAdded(N), d_vecSubd(N);
    thrust::device_vector<int> d_vecMultd(N), d_vecModed(N);

    cudaEvent_t start_time = get_time();

    // do math
    thrust::transform(d_vecA.begin(), d_vecA.end(), d_vecB.begin(), 
                        d_vecAdded.begin(), thrust::plus<int>());
    thrust::transform(d_vecA.begin(), d_vecA.end(), d_vecB.begin(), 
                        d_vecSubd.begin(), thrust::minus<int>());
    thrust::transform(d_vecA.begin(), d_vecA.end(), d_vecB.begin(), 
                        d_vecMultd.begin(), thrust::multiplies<int>());
    thrust::transform(d_vecB.begin(), d_vecB.end(), d_vecA.begin(), 
                        d_vecModed.begin(), thrust::modulus<int>());
    // transfer back to host
    thrust::copy(d_vecAdded.begin(), d_vecAdded.end(), h_vecAdd.begin());
    thrust::copy(d_vecSubd.begin(), d_vecSubd.end(), h_vecSub.begin());
    thrust::copy(d_vecMultd.begin(), d_vecMultd.end(), h_vecMult.begin());
    thrust::copy(d_vecModed.begin(), d_vecModed.end(), h_vecMod.begin());

    // end timing
    cudaEvent_t end_time = get_time();
    cudaEventSynchronize(end_time);
    float delta = 0;
    cudaEventElapsedTime(&delta, start_time, end_time);
    printf("Thrust with Pinned mem: %3.3f ms\n", delta); 
    cudaEventDestroy(start_time); cudaEventDestroy(end_time);
}
/* 
    Host (pageable) -> device, computes math output vectors, 
    device -> host, prints timing metric.
*/
void pageable_sub(int N)
{
    // generate host vecs, populate A with rand, B with sequence
    thrust::host_vector<int> h_vecA(N), h_vecB(N);
    thrust::sequence(h_vecA.begin(), h_vecA.end());
    thrust::generate(h_vecB.begin(), h_vecB.end(), rand_mod4);

    // initialize output host vectors
    thrust::host_vector<int> h_vecAdd(N), h_vecSub(N);
    thrust::host_vector<int> h_vecMult(N), h_vecMod(N);

    // transfer inputs to device, init outputs
    thrust::device_vector<int> d_vecA = h_vecA, d_vecB = h_vecB;
    thrust::device_vector<int> d_vecAdded(N), d_vecSubd(N);
    thrust::device_vector<int> d_vecMultd(N), d_vecModed(N);

    cudaEvent_t start_time = get_time();

    // do math
    thrust::transform(d_vecA.begin(), d_vecA.end(), d_vecB.begin(), 
                      d_vecAdded.begin(), thrust::plus<int>());
    thrust::transform(d_vecA.begin(), d_vecA.end(), d_vecB.begin(), 
                      d_vecSubd.begin(), thrust::minus<int>());
    thrust::transform(d_vecA.begin(), d_vecA.end(), d_vecB.begin(), 
                      d_vecMultd.begin(), thrust::multiplies<int>());
    thrust::transform(d_vecB.begin(), d_vecB.end(), d_vecA.begin(), 
                      d_vecModed.begin(), thrust::modulus<int>());
    // transfer back to host
    thrust::copy(d_vecAdded.begin(), d_vecAdded.end(), h_vecAdd.begin());
    thrust::copy(d_vecSubd.begin(), d_vecSubd.end(), h_vecSub.begin());
    thrust::copy(d_vecMultd.begin(), d_vecMultd.end(), h_vecMult.begin());
    thrust::copy(d_vecModed.begin(), d_vecModed.end(), h_vecMod.begin());

    // end timing
    cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	float delta = 0;
    cudaEventElapsedTime(&delta, start_time, end_time);
    printf("Thrust with Pageable mem: %3.3f ms\n", delta);
    cudaEventDestroy(start_time); cudaEventDestroy(end_time);
}

/* 
    Computes all math functions using either pinned or pageable host memory,
    for two sets of thread and block sizes using thrust.
*/
int main(void)
{
    // test one
    pageable_sub(N); 
    pinned_sub(N); 
    
    N *= 2;
    // test two
    pageable_sub(N); 
    pinned_sub(N); 

	return EXIT_SUCCESS;
}