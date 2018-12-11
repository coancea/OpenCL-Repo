#include "../utils/Util.h"
#include "../utils/GPU_Constants.h"

#define logWORKGROUP_SIZE   7
#define WORKGROUP_SIZE      (1<<logWORKGROUP_SIZE)
#define REPEAT              15
#define ELEMS_PER_THREAD    9

#include "GenericHack.h"
#include "../utils/SDK_stub.h"
#include "SetupOpenCL.h"

void runMemCopy(bool is_sgm, uint8_t* cpu_flg, ElTp* cpu_inp) {
    const uint32_t num_blocks     = (buffs.N + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    const size_t   localWorkSize  = WORKGROUP_SIZE;
    const size_t   globalWorkSize = num_blocks * localWorkSize;

    { // run kernel
        cl_int ciErr1 = CL_SUCCESS;

        // make two dry runs
        for (int32_t i=0; i<2; i++) { 
            ciErr1 |= clEnqueueNDRangeKernel(ctrl.cqCommandQueue, kers.mem_cpy_ker, 1, NULL,
                                             &globalWorkSize, &localWorkSize, 0, NULL, NULL);
            clFinish(ctrl.cqCommandQueue);
            oclCheckError(ciErr1, CL_SUCCESS);
        }

        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        { // timing runs
            for (int32_t i = 0; i < REPEAT; i++) {
                ciErr1 |= clEnqueueNDRangeKernel(ctrl.cqCommandQueue, kers.mem_cpy_ker, 1, NULL,
                                                 &globalWorkSize, &localWorkSize, 0, NULL, NULL);
            }
            clFinish(ctrl.cqCommandQueue);
        }

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;
        oclCheckError(ciErr1, CL_SUCCESS);

        if (is_sgm)
            fprintf(stdout, "GPU Memcopy-WithFlag runs in: %ld microseconds on GPU; N: %d\n\n", elapsed/REPEAT, buffs.N);
        else
            fprintf(stdout, "GPU Memcopy-Straight runs in: %ld microseconds on GPU; N: %d\n\n", elapsed/REPEAT, buffs.N);
    }
} 

/** Input:
 *    @is_sgm@  segmented or starightforward version of scan 
 *    @cpu_flg@ the input  flag  array allocated on the host (CPU)
 *    @cpu_inp@ the input  value array allocated on the host (CPU)
 *    The length of the input arrays is available in @buffs.N@
 *  The Result:
 *    is stored in @buffs.gpu_out@ -- see SetupOpenCL.h file. 
 */
void runSinglePassScan(bool is_sgm, uint8_t* cpu_flg, ElTp* cpu_inp) {
    const size_t   numelems_group = WORKGROUP_SIZE * getNumElemPerThread(is_sgm);
    const uint32_t num_blocks     = (buffs.N + numelems_group - 1) / numelems_group;
    const size_t   localWorkSize  = WORKGROUP_SIZE;
    const size_t   globalWorkSize = num_blocks * localWorkSize;

    { // run kernel
        cl_int ciErr1 = CL_SUCCESS;

        // make two dry runs
        for (int32_t i=0; i<2; i++) { 
            ciErr1 |= clEnqueueNDRangeKernel(ctrl.cqCommandQueue, kers.single_scan_ker, 1, NULL,
                                             &globalWorkSize, &localWorkSize, 0, NULL, NULL);
            clFinish(ctrl.cqCommandQueue);
            oclCheckError(ciErr1, CL_SUCCESS);
        }

        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        { // timing runs
            for (int32_t i = 0; i < REPEAT; i++) {
                ciErr1 |= clEnqueueNDRangeKernel(ctrl.cqCommandQueue, kers.single_scan_ker, 1, NULL,
                                                 &globalWorkSize, &localWorkSize, 0, NULL, NULL);
            }
            clFinish(ctrl.cqCommandQueue);
        }

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;
        oclCheckError(ciErr1, CL_SUCCESS);
        if (is_sgm)
            fprintf(stdout, "GPU Single-Pass Segmented-Scan runs in: %ld microseconds on GPU; N: %d ...", elapsed/REPEAT, buffs.N);
        else
            fprintf(stdout, "GPU Single-Pass Inclusive-Scan runs in: %ld microseconds on GPU; N: %d ...", elapsed/REPEAT, buffs.N);
    }
} 

void mkRandomDataset (const uint32_t N, ElTp* data, uint8_t* flags) {
    for (int i = 0; i < N; ++i) {
        float r01 = rand() / (float)RAND_MAX;
        float r   = r01 - 0.5;
        data[i]   = spreadData(r);
        flags[i]  = r01 > 0.98 ? 1 : 0; //(i % 11) == 0 ? 1 : 0;
    }
    flags[0] = 1;
}

void goldenScan (bool is_sgm, const uint32_t N, ElTp* cpu_inp, uint8_t* cpu_flags, ElTp* cpu_out) {
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int r=0; r < 1/*REPEAT*/; r++) {
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

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;
    fprintf(stdout, "Sequential golden scan runs in: %ld microseconds on CPU\n", elapsed/REPEAT);
}

void validate(const uint32_t N, ElTp* cpu_out, ElTp* gpu_out, int32_t eps_range) {
    // every eps_range elems we will allow an EPS error to propagate
    float ok_error = ((N + eps_range - 1) / eps_range) * EPS; 
    bool success = true;
    for(uint32_t i=0; i<N; i++) {
        float err = (float) (cpu_out[i] - gpu_out[i]);
        if (fabs(err) > ok_error) {
            success = false;
            fprintf(stdout, "INVALID RESULT at index: %d. Expected: %f. Actual: %f!\n", i, (float)cpu_out[i], (float)gpu_out[i]);
            break;
        }
    }
    if (success) fprintf(stdout, "VALID RESULT!\n");
}

void testOnlyScan(const uint32_t N, ElTp *cpu_inp, uint8_t* cpu_flg, ElTp *cpu_ref, ElTp *cpu_out) {
    // init buffers and kernels arguments
    initOclBuffers(N, false, cpu_flg, cpu_inp);
    initKernels(false);

    // compute sequential (golden) scan version 
    goldenScan(false, N, cpu_inp, cpu_flg, cpu_ref);

    // compute single-pass scan on GPUs
    runSinglePassScan(false, cpu_flg, cpu_inp);

    // WRITE THE RESULT ARRAY BACK TO CPU
    gpuToCpuTransfer(N, cpu_out);

    // validate GPU result
    validate(N, cpu_ref, cpu_out, 10000);

    // run memcopy kernel
    runMemCopy(false, cpu_flg, cpu_inp);

    // Release GPU Buffer/Kernels resources!!!
    oclReleaseBuffKers(false);
}

void testSegmScan(const uint32_t N, ElTp *cpu_inp, uint8_t* cpu_flg, ElTp *cpu_ref, ElTp *cpu_out) {
    // init buffers and kernels arguments
    initOclBuffers(N, true, cpu_flg, cpu_inp);
    initKernels(true);

    // compute sequential (golden) scan version 
    goldenScan(true, N, cpu_inp, cpu_flg, cpu_ref);

    // compute single-pass scan on GPUs
    runSinglePassScan(true, cpu_flg, cpu_inp);

    // WRITE THE RESULT ARRAY BACK TO CPU
    gpuToCpuTransfer(N, cpu_out);

    // validate GPU result
    validate(N, cpu_ref, cpu_out, 100000000);

    // run memcopy kernel
    runMemCopy(true, cpu_flg, cpu_inp);

    // Release GPU Buffer/Kernels resources!!!
    oclReleaseBuffKers(true);
}

int main() {
    const uint32_t N = 100000000;

    // allocate and CPU arrays and initialize   
    ElTp* cpu_inp = (ElTp*)malloc(N*sizeof(ElTp));
    ElTp* cpu_ref = (ElTp*)malloc(N*sizeof(ElTp));
    ElTp* cpu_out = (ElTp*)malloc(N*sizeof(ElTp));
    uint8_t* cpu_flg = (uint8_t*)malloc(N);

    mkRandomDataset(N, cpu_inp, cpu_flg);

    {
        bool sanity_dev_id = (GPU_DEV_ID >= 0) && (GPU_DEV_ID < 16);
        assert(sanity_dev_id && "GPU DEVICE ID < 0 !\n");
        ctrl.dev_id = GPU_DEV_ID;
        initOclControl();
    }

    testOnlyScan(N/1000, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testSegmScan(N/1000, cpu_inp, cpu_flg, cpu_ref, cpu_out);

    testOnlyScan(N/100, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testSegmScan(N/100, cpu_inp, cpu_flg, cpu_ref, cpu_out);

    testOnlyScan(N/10, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testSegmScan(N/10, cpu_inp, cpu_flg, cpu_ref, cpu_out);

    testOnlyScan(N, cpu_inp, cpu_flg, cpu_ref, cpu_out);
    testSegmScan(N, cpu_inp, cpu_flg, cpu_ref, cpu_out);

    oclControlCleanUp();
    free(cpu_inp);
    free(cpu_flg);
    free(cpu_ref);
    free(cpu_out);
}

// /usr/lib/x86_64-linux-gnu/libOpenCL.so
