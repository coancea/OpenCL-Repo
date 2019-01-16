#include "../../clutils.h"
#include <math.h>

#define NUM_GROUPS_SCAN     1024
#define WORKGROUP_SIZE      256
#define RUNS_GPU            300
#define ELEMS_PER_THREAD    7

#include "bridge.h"
#include "helper.h"
#include "scan.h"

void testIncScan(const uint32_t N, ElTp *cpu_inp, uint8_t* cpu_flg, ElTp *cpu_ref, ElTp *cpu_out) {
    // init buffers and kernels arguments
    initOclBuffers(N, cpu_flg, cpu_inp);
    initKernels();

    // compute sequential (golden) scan version 
    goldenScan(0, N, cpu_inp, cpu_flg, cpu_ref);

    // run memcopy kernel
    profileMemcpy();

    { // inclusive scan on GPU
        IncScanBuffs arrs;
        arrs.N   = N;
        arrs.inp = buffs.inp;
        arrs.out = buffs.out;
        arrs.tmp = buffs.tmp_val;
        profileScan(arrs, cpu_ref, cpu_out);
    }

    //printf("\n");
    // compute sequential (golden) segmented scan version 
    goldenScan(1, N, cpu_inp, cpu_flg, cpu_ref);

    { // segmented inclusive scan on GPU
        SgmScanBuffs arrs;
        arrs.N   = N;
        arrs.inp = buffs.inp;
        arrs.flg = buffs.flg;
        arrs.out = buffs.out;
        arrs.tmp_val = buffs.tmp_val;
        arrs.tmp_flg = buffs.tmp_flg;
        profileSgmScan(arrs, cpu_ref, cpu_out);
    }
    printf("\n\n");

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

    testIncScan(N/512, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testIncScan(N/64, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testIncScan(N/8, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testIncScan(N, cpu_inp, cpu_flg, cpu_ref, cpu_out);

    freeOclControl();
    free(cpu_inp);
    free(cpu_flg);
    free(cpu_ref);
    free(cpu_out);
}

// /usr/lib/x86_64-linux-gnu/libOpenCL.so
// For Apple: g++ -O2 Scan.cpp -framework OpenCL
