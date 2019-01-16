#include "../../clutils.h"
#include <math.h>

#define NUM_GROUPS_SCAN     256
#define WORKGROUP_SIZE      256
#define RUNS_GPU            200
#define ELEMS_PER_THREAD    7

#include "bridge.h"
#include "helper.h"
#include "scan.h"

void testOnlyScan(const uint32_t N, ElTp *cpu_inp, uint8_t* cpu_flg, ElTp *cpu_ref, ElTp *cpu_out) {
    // init buffers and kernels arguments
    initOclBuffers(N, cpu_flg, cpu_inp);
    initKernels();

    // compute sequential (golden) scan version 
    goldenScan(0, N, cpu_inp, cpu_flg, cpu_ref);

    // run memcopy kernel
    profileMemcpy();

    { // compute single-passscan on GPUs
        IncScanBuffs arrs;
        arrs.N   = N;
        arrs.inp = buffs.inp;
        arrs.out = buffs.out;
        arrs.tmp = buffs.tmp_val;
        profileScan(arrs, cpu_ref, cpu_out);
    }

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
    uint8_t* cpu_flg = (uint8_t*)malloc(N);

    mkRandomDataset(N, cpu_inp, cpu_flg);
    initOclControl();

    testOnlyScan(N/512, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testOnlyScan(N/64, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testOnlyScan(N/8, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testOnlyScan(N, cpu_inp, cpu_flg, cpu_ref, cpu_out);

    freeOclControl();
    free(cpu_inp);
    free(cpu_flg);
    free(cpu_ref);
    free(cpu_out);
}

// /usr/lib/x86_64-linux-gnu/libOpenCL.so
// For Apple: g++ -O2 Scan.cpp -framework OpenCL
