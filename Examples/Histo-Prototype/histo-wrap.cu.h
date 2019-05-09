#ifndef HISTO_WRAPPER
#define HISTO_WRAPPER

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

void randomInit(int* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand(); // (float)RAND_MAX;
}

void zeroOut(int* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = 0;
}

template<class T>
bool validate(T* A, T* B, unsigned int sizeAB) {
    for(int i = 0; i < sizeAB; i++) {
        if (A[i] != B[i]) {
            printf("INVALID RESULT %d %d %d\n", i, A[i], B[i]);
            return false;
        }
    }
    //printf("VALID RESULT!\n");
    return true;
}

void computeSeqHisto(const int N, const int H, int* input, int* histo) {
    for(int i=0; i<N; i++) {
        struct indval<int> iv = f<int>(input[i], H);
        histo[iv.index] += iv.value;
    }
}

unsigned long
goldSeqHisto(const int N, const int H, int* input, int* histo) {
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int q=0; q<CPU_RUNS; q++) {
        zeroOut(histo, H);
        computeSeqHisto(N, H, input, histo);
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    //printf("Sequential Naive version runs in: %lu microsecs\n", elapsed);
    return (elapsed / CPU_RUNS); 
}

unsigned long
locMemHwdAddCoop(AtomicPrim select, const int N, const int H, const int histos_per_block, int* d_input, int* h_ref_histo) {
    if(histos_per_block <= 0) {
        printf("Illegal subhistogram degree: %d, H:%d, XCG?=%d, EXITING!\n", histos_per_block, H, (select==XCHG));
        exit(0);
    }

    // setup execution parameters
    const size_t num_blocks = (NUM_THREADS(N) + BLOCK - 1) / BLOCK; 
    const size_t shmem_size = histos_per_block * H * sizeof(int);

    const size_t mem_size_histo  = H * sizeof(int);
    const size_t mem_size_histos = histos_per_block * num_blocks * mem_size_histo;
    int* d_histos;
    int* d_histo;
    int* h_histo = (int*)malloc(H*sizeof(int));
    cudaMalloc((void**) &d_histos, mem_size_histos);
    cudaMalloc((void**) &d_histo,  H * sizeof(int));

    //printf("multi-histogram degree: %d, H:%d\n", histos_per_block, H);

    { // dry run
      if(select == ADD) {
        locMemHwdAddCoopKernel<ADD><<< num_blocks, BLOCK, shmem_size >>>
            (N, H, histos_per_block, NUM_THREADS(N), d_input, d_histos);
      } else if (select == CAS){
        locMemHwdAddCoopKernel<CAS><<< num_blocks, BLOCK, shmem_size >>>
            (N, H, histos_per_block, NUM_THREADS(N), d_input, d_histos);
      } else { // select == XCHG
        locMemHwdAddCoopKernel<XCHG><<< num_blocks, BLOCK, 2*shmem_size >>>
            (N, H, histos_per_block, NUM_THREADS(N), d_input, d_histos);
      }
    }

    cudaMemset(d_histos, 0, mem_size_histos);
    cudaMemset(d_histo , 0, mem_size_histo );

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    for(int q=0; q<GPU_RUNS; q++) {
      //cudaMemset(d_histos, 0, mem_size_histos);
      if(select == ADD) {
        locMemHwdAddCoopKernel<ADD><<< num_blocks, BLOCK, shmem_size >>>
            (N, H, histos_per_block, NUM_THREADS(N), d_input, d_histos);
      } else if (select == CAS){
        locMemHwdAddCoopKernel<CAS><<< num_blocks, BLOCK, shmem_size >>>
            (N, H, histos_per_block, NUM_THREADS(N), d_input, d_histos);
      } else { // select == XCHG
        locMemHwdAddCoopKernel<XCHG><<< num_blocks, BLOCK, 2*shmem_size >>>
            (N, H, histos_per_block, NUM_THREADS(N), d_input, d_histos);
      }
    }
    cudaThreadSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

    { // reduce across histograms and copy to host
        const size_t num_blocks_red = (H + BLOCK - 1) / BLOCK;
        naive_reduce_kernel<<< num_blocks_red, BLOCK >>>(d_histos, d_histo, H, histos_per_block*num_blocks);
        cudaMemcpy(h_histo, d_histo, mem_size_histo, cudaMemcpyDeviceToHost);
    }

    { // validate and free memory
        bool is_valid = validate<int>(h_histo, h_ref_histo, H);

        free(h_histo);
        cudaFree(d_histos);
        cudaFree(d_histo);

        if(!is_valid) {
            printf("locMemHwdAddCoop: Validation FAILS! Exiting!\n\n");
            exit(1);
        }
    }

    return (elapsed/GPU_RUNS);
}

#endif // HISTO_WRAPPER
