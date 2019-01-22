#include "../../clutils.h"
#include <math.h>

typedef int32_t     ElTp;
#define ElTp_STR    "int32_t"
#define NE          0

inline uint32_t pred(int32_t k) {
    return (1 - (k & 1));
}

#define lgWARP              5
#define WARP                (1<<lgWARP)

#define NUM_GROUPS_SCAN     1024
#define WORKGROUP_SIZE      256
#define ELEMS_PER_THREAD    9
#define RUNS_GPU            300


#include "partition2.h"

void testOptPartition(const uint32_t N, ElTp *cpu_inp, ElTp *cpu_ref, ElTp *cpu_out) {
    // init buffers and kernels arguments
    initOclBuffers(N, cpu_inp);
    initKernels();

    // compute sequential (golden) scan version 
    goldenPartition(N, cpu_inp, cpu_ref);

    // inclusive scan on GPU
    profilePartition(buffs, cpu_ref, cpu_out);

    printf("\n");

    // Release GPU Buffer/Kernels resources!!!
    freeOclBuffKers();
}

int main() {
    const uint32_t N = 4096*4096*4 + 5555;

    // allocate and CPU arrays and initialize   
    ElTp* cpu_inp = (ElTp*)malloc(N*sizeof(ElTp));
    ElTp* cpu_ref = (ElTp*)malloc(N*sizeof(ElTp));
    ElTp* cpu_out = (ElTp*)malloc(N*sizeof(ElTp));

    mkRandomDataset(N, cpu_inp);
    initOclControl();

    testOptPartition(N/512, cpu_inp, cpu_ref, cpu_out);
    testOptPartition(N/64, cpu_inp, cpu_ref, cpu_out);
    testOptPartition(N/8, cpu_inp, cpu_ref, cpu_out);
    testOptPartition(N, cpu_inp, cpu_ref, cpu_out);

    freeOclControl();
    free(cpu_inp);
    free(cpu_ref);
    free(cpu_out);
}
