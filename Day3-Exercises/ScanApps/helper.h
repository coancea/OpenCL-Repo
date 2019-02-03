#ifndef HELPER
#define HELPER

typedef struct OCLControl {
    cl_context          ctx;            // OpenCL context
    cl_device_id        device;      // OpenCL device list
    cl_program          prog;      // OpenCL program
    cl_command_queue    queue; // command queue of the targe GPU device
} OclControl;

typedef struct OCLKernels {
    cl_kernel memCpy; // Memcopy Kernel

    /** scan and segmented scan kernels **/
    cl_kernel grpRed;      // reduction-phase kernel for straight scan
    cl_kernel grpSgmRed;   // reduction-phase kernel for segmented scan

    cl_kernel shortScan;
    cl_kernel shortSgmScan;

    cl_kernel grpScan;     // inclusive-scan kernel
    cl_kernel grpSgmScan;  // segmented-scan kernel (also inclusive)

    /** partition2 kernels **/
    cl_kernel mapPredPart;     // partition2 kernel
    cl_kernel scatterPart;     // partition2 kernel

    /** sparse-matrix-vector multiplication kernels **/
    cl_kernel iniFlagsSpMVM;
    cl_kernel mkFlagsSpMVM;
    cl_kernel mulPhaseSpMVM;
    cl_kernel getLastSpMVM;
} OclKernels;

typedef struct OCLBuffers {
    // constants
    uint32_t N;

    // main input and result arrays (global memory)
    cl_mem  inp; // length `N` elements
    cl_mem  flg; // length `N` bytes
    cl_mem  out; // length `N` elements

    cl_mem  tmp_val; // holds `NUM_GROUPS_SCAN` elements
    cl_mem  tmp_flg; // holds `NUM_GROUPS_SCAN` bytes
} OclBuffers;

//CpuArrays  arrs;
OclControl ctrl;
OclKernels kers;
OclBuffers buffs;

void cleanUpBuffer(size_t buf_len, cl_mem buf);

void initOclControl() {
    char    compile_opts[128];
    sprintf(compile_opts, "-D lgWARP=%d -D ELEMS_PER_THREAD=%d -D NE=%d -D ElTp=%s",
            lgWARP, ELEMS_PER_THREAD, NE, ElTp_STR);
    
    //opencl_init_command_queue(0, GPU_DEV_ID, &ctrl.device, &ctrl.ctx, &ctrl.queue);
    opencl_init_command_queue_default(&ctrl.device, &ctrl.ctx, &ctrl.queue);
    ctrl.prog = opencl_build_program(ctrl.ctx, ctrl.device, "scanapps.cl", compile_opts);
}

void initOclBuffers(const uint32_t N, uint8_t* cpu_flg, ElTp* cpu_inp) {
    cl_int error = CL_SUCCESS;
    size_t size;

    // constants
    buffs.N = N;

    // global-memory input buffer
    size = N*sizeof(ElTp);
    buffs.inp = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, cpu_inp, &error);
    OPENCL_SUCCEED(error);

    // global-memory flags buffer
    size = N*sizeof(uint8_t);
    buffs.flg = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, cpu_flg, &error);
    OPENCL_SUCCEED(error);

    // global-memory scan result
    size = N*sizeof(ElTp);
    buffs.out = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error);
    OPENCL_SUCCEED(error);

    // temporary value and flag buffers for the one-block scan
    size = NUM_GROUPS_SCAN * sizeof(ElTp);
    buffs.tmp_val = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error);
    OPENCL_SUCCEED(error);

    size = NUM_GROUPS_SCAN * sizeof(uint8_t);
    buffs.tmp_flg = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error);
    OPENCL_SUCCEED(error);
}

void initKernels() {
    cl_int error;

    kers.memCpy = clCreateKernel(ctrl.prog, "memcpy_simple", &error);
    OPENCL_SUCCEED(error);

    kers.grpRed = clCreateKernel(ctrl.prog, "redPhaseKer", &error);
    OPENCL_SUCCEED(error);

    kers.grpSgmRed = clCreateKernel(ctrl.prog, "redPhaseSgmKer", &error);
    OPENCL_SUCCEED(error);

    kers.shortScan = clCreateKernel(ctrl.prog, "shortScanKer", &error);
    OPENCL_SUCCEED(error);

    kers.shortSgmScan = clCreateKernel(ctrl.prog, "shortSgmScanKer", &error);
    OPENCL_SUCCEED(error);

    kers.grpScan = clCreateKernel(ctrl.prog, "scanPhaseKer", &error);
    OPENCL_SUCCEED(error);

    kers.grpSgmScan = clCreateKernel(ctrl.prog, "scanPhaseSgmKer", &error);
    OPENCL_SUCCEED(error);

    kers.mapPredPart= clCreateKernel(ctrl.prog, "mapPredPartKer", &error);
    OPENCL_SUCCEED(error);

    kers.scatterPart= clCreateKernel(ctrl.prog, "scatterPartKer", &error);
    OPENCL_SUCCEED(error);

    kers.iniFlagsSpMVM= clCreateKernel(ctrl.prog, "iniFlagsSpMVM", &error);
    OPENCL_SUCCEED(error);

    kers.mkFlagsSpMVM= clCreateKernel(ctrl.prog, "mkFlagsSpMVM", &error);
    OPENCL_SUCCEED(error);

    kers.mulPhaseSpMVM= clCreateKernel(ctrl.prog, "mulPhaseSpMVM", &error);
    OPENCL_SUCCEED(error);

    kers.getLastSpMVM= clCreateKernel(ctrl.prog, "getLastSpMVM", &error);
    OPENCL_SUCCEED(error);
}

void gpuToCpuTransfer(const uint32_t N, cl_mem gpu_out, ElTp* cpu_out) {
    cl_int  error = clEnqueueReadBuffer (
                        ctrl.queue, gpu_out, CL_TRUE, 0,
                        N*sizeof(ElTp), cpu_out, 0, NULL, NULL
                    );
    OPENCL_SUCCEED(error);
}

void freeOclBuffKers() {
    //fprintf(stderr, "Releasing Kernels...\n");
    clReleaseKernel(kers.memCpy);
    clReleaseKernel(kers.grpRed);
    clReleaseKernel(kers.shortScan);
    clReleaseKernel(kers.grpScan);
    clReleaseKernel(kers.grpSgmRed);
    clReleaseKernel(kers.shortSgmScan);
    clReleaseKernel(kers.grpSgmScan);

    clReleaseKernel(kers.mapPredPart);
    clReleaseKernel(kers.scatterPart);
    clReleaseKernel(kers.iniFlagsSpMVM);
    clReleaseKernel(kers.mkFlagsSpMVM);
    clReleaseKernel(kers.mulPhaseSpMVM);
    clReleaseKernel(kers.getLastSpMVM);

    //fprintf(stderr, "Releasing GPU buffers ...\n");
    clReleaseMemObject(buffs.inp);
    clReleaseMemObject(buffs.flg);
    clReleaseMemObject(buffs.out);
    clReleaseMemObject(buffs.tmp_val);
    clReleaseMemObject(buffs.tmp_flg);
}

void freeOclControl() {
    //fprintf(stderr, "Releasing GPU program ...\n");
    clReleaseProgram(ctrl.prog);

    //fprintf(stderr, "Releasing Command Queue ...\n");
    clReleaseCommandQueue(ctrl.queue);
        
    //fprintf(stderr, "Releasing GPU context ...\n");
    clReleaseContext(ctrl.ctx);
}

/////////////////////////////////
// various utility functions
/////////////////////////////////

void validate(ElTp* A, ElTp* B, uint32_t sizeAB){
    for(uint32_t i = 0; i < sizeAB; i++)
      if (A[i] != B[i]) {
        printf("INVALID RESULT %d %d %d\n", i, A[i], B[i]);
        return;
      }
    printf("VALID RESULT!\n");
}

//inline ElTp spreadData(float r) { return (ElTp)r; }
inline ElTp spreadData(float r) { return (ElTp)(r * 10.0); }

void mkRandomDataset (const uint32_t N, ElTp* data, uint8_t* flags) {
    for (uint32_t i = 0; i < N; ++i) {
        float r01 = rand() / (float)RAND_MAX;
        float r   = r01 - 0.5;
        data[i]   = spreadData(r);
        flags[i]  = r01 > 0.95 ? 1 : 0;
    }
    flags[0] = 1;
}

////////////////////////////////
// memcpy kernel
////////////////////////////////

inline size_t mkGlobalDim(const uint32_t pardim, const uint32_t T) {
    return ((pardim + T - 1) / T) * T;
}

cl_int runMemcpy() {
    cl_int error = CL_SUCCESS;
    const size_t   localWorkSize  = WORKGROUP_SIZE;
    const size_t   globalWorkSize = mkGlobalDim(buffs.N, WORKGROUP_SIZE);

    error |= clSetKernelArg(kers.memCpy, 0, sizeof(uint32_t), (void *)&buffs.N);
    error |= clSetKernelArg(kers.memCpy, 1, sizeof(cl_mem), (void*)&buffs.inp);
    error |= clSetKernelArg(kers.memCpy, 2, sizeof(cl_mem), (void*)&buffs.out);

    error |= clEnqueueNDRangeKernel(ctrl.queue, kers.memCpy, 1, NULL,
                                    &globalWorkSize, &localWorkSize, 0, NULL, NULL);

    return error;
}

void profileMemcpy() {
    int64_t elapsed, aft, bef;
    cl_int error = CL_SUCCESS;

    // make a dry run
    error |= runMemcpy();
    clFinish(ctrl.queue);
    OPENCL_SUCCEED(error);

    // timing runs
    bef = get_wall_time();
    for (int32_t i = 0; i < RUNS_GPU; i++) {
        error |= runMemcpy();
    }
    clFinish(ctrl.queue);
    
    aft = get_wall_time();
    elapsed = aft - bef;
    OPENCL_SUCCEED(error);

    {
        double microsecPerTransp = ((double)elapsed)/RUNS_GPU; 
        double bytesaccessed = 2 * buffs.N * sizeof(ElTp); // one read + one write
        double gigaBytesPerSec = (bytesaccessed * 1.0e-3f) / microsecPerTransp;

        printf("GPU memCpy Straight (Ideal) runs in: %ld microseconds on GPU; N: %d, GBytes/sec: %.2f\n",
               elapsed/RUNS_GPU, buffs.N, gigaBytesPerSec);
    }
}

void cleanUpBuffer(size_t buf_len, cl_mem buf) {
    ElTp pattern = 0.0;
    cl_int error = 
        clEnqueueFillBuffer( ctrl.queue, buf, (void*)&pattern, sizeof(pattern),
                             0, buf_len*sizeof(pattern), 0, NULL, NULL );
    clFinish(ctrl.queue);
    OPENCL_SUCCEED(error);
}

#endif //HELPER
