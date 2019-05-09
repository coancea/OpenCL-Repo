#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 

#define BLOCK       1024
#define GPU_RUNS    100
#define CPU_RUNS    1

#define INP_LEN     50000000
#define Hmax        100000

#define RACE_FACT   8

#define DEBUG_INFO  0

cudaDeviceProp prop;
unsigned int HWD;
unsigned int SH_MEM_SZ;
unsigned int BLOCK_SZ;

#define NUM_THREADS(n)  min(n, HWD)

#include "histo-kernels.cu.h"
#include "histo-wrap.cu.h"


int optimSubHistoDeg(const AtomicPrim prim_kind, const int Q, const int H) {
    const int el_size = (prim_kind == XCHG)? 2*sizeof(int) : sizeof(int);
    const int m = ((Q*4 / el_size) * BLOCK) / H;
    const int coop = (BLOCK + m - 1) / m;
    printf("COOP LEVEL: %d, subhistogram degree: %d\n", coop, m);
    return m;
}


void runLocalMemDataset(int* h_input, int* h_histo, int* d_input) {
    const int num_histos = 5;
    const int num_m_degs = 5;
    const int histo_sizes[num_histos] = { 31, 63, 127, 255, 511 }; //{ 64, 128, 256, 512 };
    //const AtomicPrim atomic_kinds[3] = {ADD, CAS, XCHG};

    unsigned long runtimes[3][num_histos][num_m_degs];

    for(int i=0; i<num_histos; i++) {
        const int H = histo_sizes[i];
        const int m_opt = optimSubHistoDeg(ADD, 12, H);

        const int min_HB = min(H,BLOCK);
        const int subhisto_degs[5] = { m_opt, (8*BLOCK) / min_HB, (4*BLOCK) / min_HB, (1*BLOCK) / min_HB, 1};

        goldSeqHisto(INP_LEN, H, h_input, h_histo);

        for(int j=0; j<num_m_degs; j++) {
            const int histos_per_block = subhisto_degs[j];
            runtimes[0][i][j] = locMemHwdAddCoop(ADD, INP_LEN, H, histos_per_block, d_input, h_histo);
            runtimes[1][i][j] = locMemHwdAddCoop(CAS, INP_LEN, H, histos_per_block, d_input, h_histo);
            runtimes[2][i][j] = locMemHwdAddCoop(XCHG, INP_LEN, H, max(histos_per_block/2,1), d_input, h_histo);
        }
    }
        
    for(int k=0; k<3; k++) {
        if     (k==0) printf("LOC_ADD\t");
        else if(k==1) printf("LOC_CAS\t");
        else if(k==2) printf("LOC_XCG\t");

        for(int i = 0; i<num_histos; i++) { printf("H=%d\t", histo_sizes[i]); }
        printf("\n");
        for(int j=0; j<num_m_degs; j++) {
            printf("SH_DEG_%d\t", j);
            for(int i = 0; i<num_histos; i++) {
                printf("%lu\t", runtimes[k][i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
int main() {
    // set seed for rand()
    srand(2006);

    { // 0. querry the hardware
        int nDevices;
        cudaGetDeviceCount(&nDevices);
  
        cudaGetDeviceProperties(&prop, 0);
        HWD = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
        BLOCK_SZ = prop.maxThreadsPerBlock;
        SH_MEM_SZ = prop.sharedMemPerBlock;
        if (DEBUG_INFO) {
            printf("Device name: %s\n", prop.name);
            printf("Number of hardware threads: %d\n", HWD);
            printf("Block size: %d\n", BLOCK_SZ);
            printf("Shared memory size: %d\n", SH_MEM_SZ);
            puts("====");
        }
    }

 
    // 1. allocate host memory for input and histogram
    const unsigned int mem_size_input = sizeof(int) * INP_LEN;
    int* h_input = (int*) malloc(mem_size_input);
    const unsigned int mem_size_histo = sizeof(int) * Hmax;
    int* h_histo = (int*) malloc(mem_size_histo);
 
    // 2. initialize host memory
    randomInit(h_input, INP_LEN);
    zeroOut(h_histo, Hmax);
    
    // 3. allocate device memory for input and copy from host
    int* d_input;
    cudaMalloc((void**) &d_input, mem_size_input);
    cudaMemcpy(d_input, h_input, mem_size_input, cudaMemcpyHostToDevice);
 
#if 0
    { // 5. compute a bunch of histograms
        const int H = 128;
        
        unsigned long tm_seq = goldSeqHisto(INP_LEN, H, h_input, h_histo);
        printf("Histogram Sequential        took: %lu microsecs\n", tm_seq);

        int histos_per_block = BLOCK/32;
        //int histos_per_block = optimSubHistoDeg(CAS, 12, H); 
        unsigned long tm_add = locMemHwdAddCoop(ADD, INP_LEN, H, histos_per_block, d_input, h_histo);
        printf("Histogram Local-Mem ADD with subhisto-degree %d took: %lu microsecs\n", histos_per_block, tm_add);

        unsigned long tm_cas = locMemHwdAddCoop(CAS, INP_LEN, H, histos_per_block, d_input, h_histo);
        printf("Histogram Local-Mem CAS with subhisto-degree %d took: %lu microsecs\n", histos_per_block, tm_cas);

        //coop = optimalCoop(XCHG, 12, H);
        unsigned long tm_xch = locMemHwdAddCoop(XCHG, INP_LEN, H, histos_per_block/2, d_input, h_histo);
        printf("Histogram Local-Mem XCG with subhisto-degree %d took: %lu microsecs\n", histos_per_block, tm_xch);
    }
#endif

#if 1
    runLocalMemDataset(h_input, h_histo, d_input);
#endif

    // 7. clean up memory
    free(h_input);
    free(h_histo);
    cudaFree(d_input);
}

