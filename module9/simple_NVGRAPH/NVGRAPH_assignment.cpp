#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "nvgraph.h"

__host__ cudaEvent_t get_time(void) {
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}
/*
 * Allocates pageable memory for host's input and output data
 */
void pageableAlloc(int n_nodes, int n_edges, int **offsets, int **indices, 
                   float **weights, float **path_vals, cudaDataType_t **vertex_dimT,
                   nvgraphCSRTopology32I_t *CSR_input)
{
    //allocate
    // edge weights
    float w[] = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0};
    // first edge for each vertex
    int off[] = {0, 1, 3, 1, 5, 7, 9, 5};
    // each edge's destination vertex 
    int inds[] = {1, 0, 1, 0, 3, 5, 3, 2, 2, 3};
    float *pv; pv = (float*)malloc(n_nodes*sizeof(float));
    //the graph struct and vertices
    cudaDataType_t *v_dimT;
    v_dimT = (cudaDataType_t*)malloc(sizeof(cudaDataType_t));
    nvgraphCSRTopology32I_t csr_in;
    csr_in = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSRTopology32I_st));
    v_dimT[0] = CUDA_R_32F;
    csr_in->nvertices = n_nodes; 
    csr_in->nedges = n_edges; 
    csr_in->destination_indices = inds;
    csr_in->source_offsets = off;

    // update pointers
    *offsets = off;
    *indices = inds;
    *weights = w;
    *path_vals = pv;
    *vertex_dimT = v_dimT;
    *CSR_input = csr_in;
}

/*
 * Allocates pinned memory for host's input and output data
 */
void pinnedAlloc(int n_nodes, int n_edges, int **offsets, int **indices, 
                   float **weights, float **path_vals, cudaDataType_t **vertex_dimT,
                   nvgraphCSRTopology32I_t *CSR_input)
{
    //allocate
    // edge weights
    float w[] = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0};
    // first edge for each vertex
    int off[] = {0, 1, 3, 1, 5, 7, 9, 5};
    // each edge's destination vertex 
    int inds[] = {1, 0, 1, 0, 3, 5, 3, 2, 2, 3};
    float *pv; 
    cudaHostAlloc((void**)&pv, n_nodes*sizeof(float), cudaHostAllocDefault);
    //the graph struct and vertices
    cudaDataType_t *v_dimT;
    cudaHostAlloc((void**)&v_dimT, sizeof(cudaDataType_t), cudaHostAllocDefault);
    nvgraphCSRTopology32I_t csr_in;
    cudaHostAlloc((void**)&csr_in, sizeof(struct nvgraphCSRTopology32I_st), cudaHostAllocDefault);

    v_dimT[0] = CUDA_R_32F;
    csr_in->nvertices = n_nodes; 
    csr_in->nedges = n_edges; 
    csr_in->destination_indices = inds;
    csr_in->source_offsets = off;

    // update pointers
    *offsets = off;
    *indices = inds;
    *weights = w;
    *path_vals = pv;
    *vertex_dimT = v_dimT;
    *CSR_input = csr_in;
}

/*
 * Uses NVGraph's nvgraphWidestPath to compute the path with the largest bandwidth
 * from a source node to all other nodes in a graph. Solves the bottleneck
 * problem (i.e how do you get from source to sink s.t. the smallest
 * edge weight in the path is maximized). Adapted from provided nvgraph_Pagerank.cpp.
 */

void widestPath(int use_pinned) {
    const size_t n_nodes = 7, n_edges = 10; 
    float *wide_path_vals, *weights;
    int *source_offsets, *destination_indices;
    // initialize the graph and handle
    nvgraphHandle_t handle; nvgraphGraphDescr_t csr_graph;
    nvgraphCreate(&handle); nvgraphCreateGraphDescr (handle, &csr_graph); 
    cudaDataType_t edge_dimT = CUDA_R_32F; 
    cudaDataType_t *vertex_dimT; nvgraphCSRTopology32I_t CSR_input;

    // Init host data
    use_pinned ? pinnedAlloc(n_nodes, n_edges, &source_offsets, 
              &destination_indices, &weights, &wide_path_vals,  
              &vertex_dimT, &CSR_input)
              : pageableAlloc(n_nodes, n_edges, &source_offsets, 
              &destination_indices, &weights, &wide_path_vals,  
              &vertex_dimT, &CSR_input);

    cudaEvent_t start_time = get_time();
    // define graph on device, transfer host -> device
    nvgraphSetGraphStructure(handle, csr_graph, (void*)CSR_input, NVGRAPH_CSR_32);
    nvgraphAllocateVertexData(handle, csr_graph, 1, vertex_dimT);
    nvgraphAllocateEdgeData (handle, csr_graph, 1, &edge_dimT); 
    nvgraphSetEdgeData(handle, csr_graph, (void*)weights, 0);

    // Find widest path from source node to all other graph nodes
    int source_node = 3;
    nvgraphWidestPath(handle, csr_graph, 0, &source_node, 0);
    // Transfer device -> host
    nvgraphGetVertexData(handle, csr_graph, (void*)wide_path_vals, 0);
    // end timing
    cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	float delta = 0;
    cudaEventElapsedTime(&delta, start_time, end_time);
    use_pinned ? printf("NVGraph with pinned mem: %3.3f ms\n", delta) 
               : printf("NVGraph with pageable mem: %3.3f ms\n", delta); 
    //clean up
    cudaFreeHost(wide_path_vals); cudaFreeHost(vertex_dimT); cudaFreeHost(CSR_input);
    nvgraphDestroyGraphDescr(handle, csr_graph); nvgraphDestroy(handle);
}
int main() {
    int use_pinned = 1;
    widestPath(use_pinned);
    widestPath(!use_pinned);
    return EXIT_SUCCESS;
}



