#include "../../../cub-1.8.0/cub/cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>
#include "redbykey-helper.cu.h"

#define GPU_RUNS    200

/**
template<typename KeyT , typename ValueT >
static CUB_RUNTIME_FUNCTION cudaError_t cub::DeviceRadixSort::SortPairs	(	void * 	d_temp_storage,
size_t & 	temp_storage_bytes,
const KeyT * 	d_keys_in,
KeyT * 	d_keys_out,
const ValueT * 	d_values_in,
ValueT * 	d_values_out,
int 	num_items,
int 	begin_bit = 0,
int 	end_bit = sizeof(KeyT) * 8,
cudaStream_t 	stream = 0,
bool 	debug_synchronous = false 
)	

template<typename KeysInputIteratorT , typename UniqueOutputIteratorT , typename ValuesInputIteratorT , typename AggregatesOutputIteratorT , typename NumRunsOutputIteratorT , typename ReductionOpT >
CUB_RUNTIME_FUNCTION static __forceinline__ cudaError_t cub::DeviceReduce::ReduceByKey	(	void * 	d_temp_storage,
size_t & 	temp_storage_bytes,
KeysInputIteratorT 	d_keys_in,
UniqueOutputIteratorT 	d_unique_out,
ValuesInputIteratorT 	d_values_in,
AggregatesOutputIteratorT 	d_aggregates_out,
NumRunsOutputIteratorT 	d_num_runs_out,
ReductionOpT 	reduction_op,
int 	num_items,
cudaStream_t 	stream = 0,
bool 	debug_synchronous = false 
)
 */

struct FloatAdd
{
    template <typename T>
    __device__ CUB_RUNTIME_FUNCTION __forceinline__
    T operator()(const T &a, const T &b) const {
        return a + b; //(b < a) ? b : a;
    }
};

double sortRedByKeyCUB( uint32_t* data_keys_in,  float* data_vals_in
                      , float* histo
                      , const uint32_t N, const uint32_t H
) {
    uint32_t* data_keys_out;
    float*    data_vals_out;
    uint32_t* unique_keys;
    uint32_t* num_segments;

    { // allocating stuff
        cudaMalloc ((void**) &data_keys_out, N * sizeof(uint32_t));
        cudaMalloc ((void**) &data_vals_out, N * sizeof(float));
        cudaMalloc ((void**) &unique_keys,   H * sizeof(uint32_t));
        cudaMalloc ((void**) &num_segments,  sizeof(uint32_t));
    }

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortPairs	( tmp_sort_mem, tmp_sort_len
                                        , data_keys_in, data_keys_out
                                        , data_vals_in, data_vals_out
                                        , (int)N
                                    );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }

    void * tmp_red_mem = NULL;
    size_t tmp_red_len = 0;
    FloatAdd redop;
    

    { // reduce-by-key prelude
        cub::DeviceReduce::ReduceByKey  ( tmp_red_mem, tmp_red_len
                                        , data_keys_out, unique_keys
                                        , data_vals_out, histo
                                        , num_segments, redop, (int)N
                                        );
        cudaMalloc(&tmp_red_mem, tmp_red_len);
    }

    { // one dry run
        cub::DeviceRadixSort::SortPairs	( tmp_sort_mem, tmp_sort_len
                                        , data_keys_in, data_keys_out
                                        , data_vals_in, data_vals_out
                                        , (int)N
                                        );
        cub::DeviceReduce::ReduceByKey  ( tmp_red_mem, tmp_red_len
                                        , data_keys_out, unique_keys
                                        , data_vals_out, histo
                                        , num_segments, redop, (int)N
                                        );
        cudaThreadSynchronize();
    }

    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    for(int k=0; k<GPU_RUNS; k++) {
        cub::DeviceRadixSort::SortPairs ( tmp_sort_mem, tmp_sort_len
                                        , data_keys_in, data_keys_out
                                        , data_vals_in, data_vals_out
                                        , (int)N
                                        );
        cub::DeviceReduce::ReduceByKey  ( tmp_red_mem, tmp_red_len
                                        , data_keys_out, unique_keys
                                        , data_vals_out, histo
                                        , num_segments, redop, (int)N
                                        );
    }
    cudaThreadSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

    cudaFree(tmp_sort_mem);
    cudaFree(tmp_red_mem);
    cudaFree(data_keys_out);
    cudaFree(data_vals_out);
    cudaFree(unique_keys); 
    cudaFree(num_segments);

    return elapsed;
}


int main (int argc, char * argv[]) {
    if(argc != 3) {
        printf("Expects two arguments: the image size and the histogram size! argc:%d\n", argc);
        exit(1);
    }
    const uint32_t N = atoi(argv[1]);
    const uint32_t H = atoi(argv[2]);
    printf("Computing for image size: %d and histogram size: %d\n", N, H);

    //Allocate and Initialize Host data with random values
    uint32_t* h_keys  = (uint32_t*)malloc(N*sizeof(uint32_t));
    float*    h_vals  = (float*)   malloc(N*sizeof(float));
    float*    h_histo = (float*)   malloc(H*sizeof(float));
    float*    g_histo = (float*)   malloc(H*sizeof(float));
    randomInit(h_keys, h_vals, N, H);

    { // golden sequential histogram
        double elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        histoGold(h_keys, h_vals, N, H, g_histo);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("Golden (Sequential) Float-Add Histogram runs in: %.2f microsecs\n", elapsed);
    }

    //Allocate and Initialize Device data
    uint32_t* d_keys;
    float*    d_vals;
    float*    d_histo;
    cudaMalloc ((void**) &d_keys,  N * sizeof(uint32_t));
    cudaMalloc ((void**) &d_vals,  N * sizeof(float));
    cudaMalloc ((void**) &d_histo, H * sizeof(float));
    cudaMemcpy(d_keys, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals, N * sizeof(float),    cudaMemcpyHostToDevice);

    {
        double elapsed = 
            sortRedByKeyCUB ( d_keys,  d_vals, d_histo, N, H );

        cudaMemcpy (h_histo, d_histo, H*sizeof(float), cudaMemcpyDeviceToHost);
        printf("CUB Red-By-Key Histogram ... ");
        validate(g_histo, h_histo, H);

        printf("CUB Red-By-Key Histogram runs in: %.2f microsecs\n", elapsed);
        double gigaBytesPerSec = 4 * N * sizeof(uint32_t) * 1.0e-3f / elapsed; 
        printf( "CUB Red-By-Key Histogram GBytes/sec = %.2f!\n", gigaBytesPerSec); 
    }

    // Cleanup and closing
    cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_histo);
    free(h_keys);  free(h_vals); free(g_histo); free(h_histo);

    return 0;
}
