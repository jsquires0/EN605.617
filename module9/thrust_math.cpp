#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <iostream>

int TOTAL_THREADS = 256;
int BLOCK_SIZE = 128;

int main(void)
{
    // generate random vectors on host, populate A with a sequence, B with rand
    thrust::host_vector<int> h_vecA(TOTAL_THREADS);
    thrust::host_vector<int> h_vecB(TOTAL_THREADS);
    thrust::generate(h_vecA.begin(), h_vecA.end(), rand % 4);
    thrust::sequence(h_vecB.begin(), h_vecB.end());

    // print contents of H
    for(int i = 0; i < h_vecA.size(); i++)
    {
        std::cout << "A[" << i << "] = " << h_vecA[i] << std::endl;
        std::cout << "B[" << i << "] = " << h_vecB[i] << std::endl;

    }

}