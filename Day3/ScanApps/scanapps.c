#include "../../clutils.h"
#include <math.h>

typedef int32_t     ElTp;
#define ElTp_STR    "int32_t"
#define NE          0
#define lgWARP      5
#define WARP        (1<<lgWARP)

#define NUM_GROUPS_SCAN     1024
#define WORKGROUP_SIZE      256
#define RUNS_GPU            200
#define ELEMS_PER_THREAD    7

/**
 * CODE CLONEs!!! 
 * Please change consistently in scanapps.cl as well
 */
inline ElTp binOp(ElTp v1, ElTp v2) {
    return (v1 + v2);
}
inline uint32_t pred(int32_t k) {
    return (1 - (k & 1));
}

#include "helper.h"
#include "scan.h"
#include "partition2.h"
#include "spMatVecMult.h"

void testScanApps(const uint32_t N, ElTp *cpu_inp, uint8_t* cpu_flg, ElTp *cpu_ref, ElTp *cpu_out) {
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

    // compute sequential (golden) segmented scan version 
    goldenScan(1, N, cpu_inp, cpu_flg, cpu_ref);

    SgmScanBuffs arrs;
    { // segmented inclusive scan on GPU
        arrs.N   = N;
        arrs.inp = buffs.inp;
        arrs.flg = buffs.flg;
        arrs.out = buffs.out;
        arrs.tmp_val = buffs.tmp_val;
        arrs.tmp_flg = buffs.tmp_flg;
        profileSgmScan(arrs, cpu_ref, cpu_out);
    }

    goldenPartition(N, cpu_inp, cpu_ref);

    { // partition2
        cl_int error = CL_SUCCESS;
        uint32_t size = N*sizeof(uint32_t);
        PartitionBuffs arrs;
        arrs.N = N;
        arrs.inp = buffs.inp;
        arrs.tmp = buffs.tmp_val;
        arrs.out = buffs.out;
        // need more temporary buffers
        arrs.tfs = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error);
        OPENCL_SUCCEED(error);
        arrs.ffs = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error);
        OPENCL_SUCCEED(error);
        arrs.isT = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error);
        OPENCL_SUCCEED(error);
        arrs.isF = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error);
        OPENCL_SUCCEED(error);

        profilePartition(arrs, cpu_ref, cpu_out);

        clReleaseMemObject(arrs.tfs);
        clReleaseMemObject(arrs.ffs);
        clReleaseMemObject(arrs.isT);
        clReleaseMemObject(arrs.isF);
    }

    // finally sparse-matrix vector multiplication
    profileSpMatVectMul(arrs, cpu_inp);

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

    testScanApps(N/512, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testScanApps(N/64, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testScanApps(N/8, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testScanApps(N, cpu_inp, cpu_flg, cpu_ref, cpu_out);

    freeOclControl();
    free(cpu_inp);
    free(cpu_flg);
    free(cpu_ref);
    free(cpu_out);
}

// /usr/lib/x86_64-linux-gnu/libOpenCL.so
// For Apple: g++ -O2 Scan.cpp -framework OpenCL
