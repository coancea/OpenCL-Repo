#include "../../clutils.h"
#include <stdbool.h>
#include <math.h>

#define REPEAT              800
#define ELEMS_PER_THREAD    20//11//9

#include "GenericHack.h"
#include "SetupOpenCL.h"

int64_t runMemCopy() {
    const uint32_t num_blocks     = (buffs.N + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    const size_t   localWorkSize  = WORKGROUP_SIZE;
    const size_t   globalWorkSize = num_blocks * localWorkSize;

    int64_t elapsed = -1;
    { // run kernel
        cl_int ciErr1 = CL_SUCCESS;

        // make two dry runs
        for (int32_t i=0; i<2; i++) { 
            ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.mem_cpy_ker, 1, NULL,
                                             &globalWorkSize, &localWorkSize, 0, NULL, NULL);
            clFinish(ctrl.queue);
            OPENCL_SUCCEED(ciErr1);
        } 

        int64_t aft, bef = get_wall_time();
        { // timing runs
            for (int32_t i = 0; i < REPEAT; i++) {
                ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.mem_cpy_ker, 1, NULL,
                                                 &globalWorkSize, &localWorkSize, 0, NULL, NULL);
            }
            clFinish(ctrl.queue);
        }
        aft = get_wall_time();
        elapsed = (aft - bef) / REPEAT;
        OPENCL_SUCCEED(ciErr1);
    }
    return elapsed;
} 

/** Input:
 *    the input flag array is available in @buffs.gpu_flg@
 *    the input value array is available in @buffs.gpu_inp@
 *    The length of the input arrays is available in @buffs.N@
 *  The Result:
 *    is stored in @buffs.gpu_out@ -- see SetupOpenCL.h file. 
 */
int64_t runSinglePassScan(bool do_groupvirt, int32_t num_requested_groups) {

    const size_t   numelems_group = WORKGROUP_SIZE * getNumElemPerThread();
    const uint32_t num_virtgroups = (buffs.N + numelems_group - 1) / numelems_group;

    uint32_t num_physgroups = num_virtgroups;
    if (do_groupvirt && num_requested_groups > 0)
        num_physgroups = MIN((uint32_t) num_requested_groups, num_virtgroups);

    const size_t   localWorkSize  = WORKGROUP_SIZE;
    const size_t   globalWorkSize = num_physgroups * localWorkSize;


    printf("Config:\n");
    printf("    N          = %d\n", buffs.N);
    printf("    block size = %d\n", WORKGROUP_SIZE);
    printf("    chunk      = %d\n\n", getNumElemPerThread());
    if (do_groupvirt) {
        printf("    group virtualization ON\n");
        if (num_requested_groups > 0)
            printf("    #requested groups   = %d\n", num_requested_groups);
        else
            printf("    (no num_requested_groups specified)\n");
        printf("    #virtual groups     = %d\n", num_virtgroups);
        printf("    virt factor         = %f\n", (float)num_virtgroups / num_physgroups);
    }
    else
        printf("    group virtualization OFF\n");

    printf("    #physical groups    = %d\n\n", num_physgroups);


    int64_t elapsed = -1;
    { // run kernel
        cl_int ciErr1 = CL_SUCCESS;

        // make two dry runs
        for (int32_t i=0; i<2; i++) { 
            ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.single_scan_ker, 1, NULL,
                                             &globalWorkSize, &localWorkSize, 0, NULL, NULL);
            clFinish(ctrl.queue);
            OPENCL_SUCCEED(ciErr1);
        }

        int64_t aft, bef = get_wall_time();
        { // timing runs
            for (int32_t i = 0; i < REPEAT; i++) {
                ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.single_scan_ker, 1, NULL,
                                                 &globalWorkSize, &localWorkSize, 0, NULL, NULL);
            }
            clFinish(ctrl.queue);
        }
        aft = get_wall_time();
        elapsed = (aft - bef) / REPEAT;
        OPENCL_SUCCEED(ciErr1);
    }

    return elapsed;
} 

void mkRandomDataset (const uint32_t N, ElTp* data, uint8_t* flags) {
    for (uint32_t i = 0; i < N; ++i) {
        float r01 = rand() / (float)RAND_MAX;
        float r   = r01 - 0.5;
        data[i]   = spreadData(r);
        flags[i]  = r01 > 0.98 ? 1 : 0; //(i % 11) == 0 ? 1 : 0;
    }
    flags[0] = 1;
}

int64_t goldenScan (bool is_sgm, const uint32_t N, ElTp* cpu_inp, uint8_t* cpu_flags, ElTp* cpu_out) {
    int64_t elapsed, aft, bef = get_wall_time();
    for(int r=0; r < 1; r++) {
      ElTp acc = NE;
      for(uint32_t i=0; i<N; i++) {
        if(is_sgm) {
            if (cpu_flags[i] != 0) acc = cpu_inp[i];
            else acc = binOp(acc, cpu_inp[i]);
        } else {
            acc = binOp(acc, cpu_inp[i]);
        }
        cpu_out[i] = acc;
      }
    }
    aft = get_wall_time();
    elapsed = aft - bef;
    return elapsed;
}

bool validate(const uint32_t N, ElTp* cpu_out, ElTp* gpu_out, uint32_t eps_range) {
    // every eps_range elems we will allow an EPS error to propagate
    float ok_error = ((N + eps_range - 1) / eps_range) * EPS; 
    for(uint32_t i=0; i<N; i++) {
        float err = (float) (cpu_out[i] - gpu_out[i]);
        if (fabs(err) > ok_error) {
            printf("INVALID RESULT at index: %d. Expected: %f. Actual: %f!\n",
                    i, (float)cpu_out[i], (float)gpu_out[i]);
            return false;
        }
    }
    printf("VALID RESULT!\n");
    return true;
}


void runTest(const uint32_t N, ElTp *cpu_input, uint8_t *cpu_flags, ElTp *cpu_ref, ElTp *cpu_out,
     bool is_segmented, bool do_groupvirt, int32_t num_requested_groups) {

#if 0
    if (is_segmented && do_groupvirt) {
        printf("\n\x1b[31;1mSkipping run of segmented scan with group "
               "virtualization ON, since for some\nreason it gives "
               "CL_OUT_OF_RESOURCES errors. TODO: Find out why.\n\n\x1b[0m");
        return;
    }
#endif

    printf("\n==============================================\n");
    char *s1 = is_segmented ? "Segmented" : "Regular";
    char *s2 = do_groupvirt ? "ON " : "OFF";
    printf("=== %s scan, group virtualization %-*s ===\n", s1, !is_segmented * 5, s2);
    printf("==============================================\n");


    // init buffers and kernels arguments
    initOclBuffers(N, is_segmented, cpu_flags, cpu_input);
    initKernels(is_segmented, do_groupvirt);

    // compute cpu and gpu scans
    int64_t elapsed_cpu = goldenScan(is_segmented, N, cpu_input, cpu_flags, cpu_ref);
    int64_t elapsed_gpu = runSinglePassScan(do_groupvirt, num_requested_groups);

    // write the result array back to cpu
    gpuToCpuTransfer(N, cpu_out);

    // validate GPU result
    bool valid = validate(N, cpu_ref, cpu_out, N);

    // run memcopy kernel
    int64_t elapsed_memcopy = runMemCopy();

    // Release GPU Buffer/Kernels resources!!!
    freeOclBuffKers(is_segmented);

    if (valid) {
        printf("CPU:                   %ld microsecs\n", elapsed_cpu);
        printf("GPU:                   %ld microsecs\n", elapsed_gpu);
        printf("GPU memcopy reference: %ld microsecs\n", elapsed_memcopy);
    }
    else
        printf("invalid; omitting bench results\n");
}

int main(int argc, char **argv) {

    int32_t num_requested_groups = -1;
    if (argc == 3)
        num_requested_groups = atoi(argv[2]);
    else if (argc != 2) {
        fprintf(stderr, "Usage: %s <input size> [optional #physical groups]\n",
                argv[0]);
        return 1;
    }
    int32_t N = atoi(argv[1]);

    const bool GROUPVIRT = true; // || num_requested_groups > 0;
    const bool SEGMENTED = true;

    // allocate and CPU arrays and initialize   
    ElTp* cpu_inp = (ElTp*)malloc(N*sizeof(ElTp));
    ElTp* cpu_ref = (ElTp*)malloc(N*sizeof(ElTp));
    ElTp* cpu_out = (ElTp*)malloc(N*sizeof(ElTp));
    uint8_t* cpu_flg = (uint8_t*)malloc(N);

    mkRandomDataset(N, cpu_inp, cpu_flg);
    initOclControl();

    // regular scan
    runTest(N, cpu_inp, cpu_flg, cpu_ref, cpu_out,
            !SEGMENTED, !GROUPVIRT, num_requested_groups);

    // regular scan with group virtualization
    runTest(N, cpu_inp, cpu_flg, cpu_ref, cpu_out,
             !SEGMENTED, GROUPVIRT, num_requested_groups);

    // segmented scan
    runTest(N, cpu_inp, cpu_flg, cpu_ref, cpu_out,
            SEGMENTED, !GROUPVIRT, num_requested_groups);

    // segmented scan with group virtualization
     runTest(N, cpu_inp, cpu_flg, cpu_ref, cpu_out,
             SEGMENTED, GROUPVIRT, num_requested_groups);


    freeOclControl();
    free(cpu_inp);
    free(cpu_flg);
    free(cpu_ref);
    free(cpu_out);
}

// /usr/lib/x86_64-linux-gnu/libOpenCL.so
// For Apple: g++ -O2 Scan.cpp -framework OpenCL
