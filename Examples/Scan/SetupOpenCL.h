struct OclControl {
    cl_context          cxGPUContext;          // OpenCL context
    cl_uint             nDevice;               // OpenCL gpu device count
    cl_device_id*       cdDevices;             // OpenCL device list
    cl_program          cpProgram;             // OpenCL program
    cl_int              dev_id;                // id of the target GPU device 
    cl_command_queue    cqCommandQueue;        // command queue of the targe GPU device
    cl_command_queue    cqCommandQueues[16];   // all OpenCL command queues
};

struct OclKernels {
    cl_kernel single_scan_ker = NULL;  // Single-Pass Scan Kernel
    cl_kernel mem_cpy_ker     = NULL;  // Memcopy Kernel
};

struct OclBuffers {
    // constants
    uint32_t N;
    uint32_t num_blocks_paded;

    // main input and result arrays (global memory)
    cl_mem  gpu_inp;
    cl_mem  gpu_flg;
    cl_mem  gpu_out;

    // For Single-Pass Scan: various temporary arrays
    cl_mem  global_id;
    cl_mem  aggregates; // includes incprefix
    cl_mem  incprefix;
    cl_mem  statusflgs; // includes the flags of aggregates and incprefix
};

#if 0
struct CpuArrays {
    // these are all arrays allocated in host (CPU) space 
    int32_t* cpu_inp;    // the input
    int32_t* cpu_ref;    // the (golden) result computed on CPU
    int32_t* cpu_out;    // the result computed and transfered from GPU
};
#endif

//CpuArrays  arrs;
OclControl ctrl;
OclKernels kers;
OclBuffers buffs;

int32_t getNumElemPerThread() {
    float num = (ELEMS_PER_THREAD * 4.0) / sizeof(ElTp);
    return (int32_t) num;
}

inline size_t getNumBlocks(const uint32_t N) {
    const size_t   numelems_group = WORKGROUP_SIZE * getNumElemPerThread();
    return (N + numelems_group - 1) / numelems_group;
}

uint32_t getNumBlocksPadWarp(uint32_t num_blocks, uint32_t num_bytes) {
    bool is_exact = (num_bytes > 0) && (((4*WARP) % num_bytes) == 0);
    uint32_t tau = (WARP*4) / num_bytes;
    return is_exact ? ( (num_blocks + tau - 1) / tau ) * tau : num_blocks;
}

void initOclControl() {
    int32_t num_elem_per_thread     = getNumElemPerThread();
    int32_t sgm_num_elem_per_thread = getNumElemPerThread();
    char    compile_opts[128];
    sprintf(compile_opts, "-D lgWARP=%d -D ELEMS_PER_THREAD=%d -D SGM_ELEMS_PER_THREAD=%d", 
                          lgWARP, num_elem_per_thread, sgm_num_elem_per_thread);
    build_for_GPU(
            ctrl.cxGPUContext, ctrl.cqCommandQueues, ctrl.nDevice, ctrl.cdDevices, 
            ctrl.cpProgram, ctrl.dev_id, compile_opts, "", "SinglePassScanKer"
        );
    ctrl.cqCommandQueue = ctrl.cqCommandQueues[ctrl.dev_id];
}

void initOclBuffers(const uint32_t N, bool is_sgm, uint8_t* cpu_flg, ElTp* cpu_inp) {
    cl_int ciErr, ciErr2;
    cl_int cpu_global_id[1] = {0};
    size_t size;

    // constants
    size_t  num_blocks = getNumBlocks(N);
    buffs.N = N;
    buffs.num_blocks_paded = getNumBlocksPadWarp(num_blocks, 1);

    // global-memory input buffer
    size = N*sizeof(ElTp);
    buffs.gpu_inp = clCreateBuffer(ctrl.cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, cpu_inp, &ciErr2);
    ciErr = ciErr2;

    // global-memory flags buffer
    if (is_sgm) {
        size = N*sizeof(uint8_t);
        buffs.gpu_flg = clCreateBuffer(ctrl.cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, cpu_flg, &ciErr2);
        ciErr |= ciErr2; oclCheckError(ciErr, CL_SUCCESS);
    }
    // global-memory scan result
    size = N*sizeof(ElTp);
    buffs.gpu_out = clCreateBuffer(ctrl.cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErr2);
    ciErr |= ciErr2; oclCheckError(ciErr, CL_SUCCESS);

    // global-memory intermediate-computation buffers
    {   // global_id
        buffs.global_id = clCreateBuffer(ctrl.cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(int32_t), cpu_global_id, &ciErr2);
        ciErr |= ciErr2; oclCheckError(ciErr, CL_SUCCESS);

        // allocate contiguous `aggregates` and `incprefix`
        size = num_blocks * sizeof(ElTp);
        buffs.aggregates = clCreateBuffer(ctrl.cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErr2 );
        ciErr |= ciErr2; oclCheckError(ciErr, CL_SUCCESS);
        buffs.incprefix  = clCreateBuffer(ctrl.cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErr2 );
        ciErr |= ciErr2; oclCheckError(ciErr, CL_SUCCESS);

        // allocate contiguous `status flags`, and the flags of `aggregate` and `incprefix`
        
        size = (is_sgm) ? 3 * buffs.num_blocks_paded : buffs.num_blocks_paded;
        buffs.statusflgs = clCreateBuffer(ctrl.cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErr2 );
        ciErr |= ciErr2; oclCheckError(ciErr, CL_SUCCESS);
    }
}

void initKernels(const bool is_sgm) {
    cl_int ciErr, ciErr2;
    unsigned int counter = 0;

    { // Scan kernels
        int32_t num_elem_per_thread = getNumElemPerThread();
        const size_t LOCAL_SIZE_EXCG = WORKGROUP_SIZE * num_elem_per_thread;

        kers.single_scan_ker = clCreateKernel(ctrl.cpProgram, is_sgm? "singlePassSgmScanKer" : "singlePassScanKer", &ciErr);
        oclCheckError(ciErr, CL_SUCCESS);

        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(uint32_t), (void *)&buffs.N);
        if (is_sgm) {
            ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(uint32_t), (void *)&buffs.num_blocks_paded);
            ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void*)&buffs.gpu_flg); // global flags
        }
        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void*)&buffs.gpu_inp); // global input
        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void*)&buffs.gpu_out); // global output
        oclCheckError(ciErr, CL_SUCCESS);

        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void *)&buffs.global_id );
        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void *)&buffs.aggregates);
        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void *)&buffs.incprefix );
        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void *)&buffs.statusflgs);
        oclCheckError(ciErr, CL_SUCCESS);

        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(int32_t), NULL); // __local block_id
        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, LOCAL_SIZE_EXCG * sizeof(ElTp), NULL); // local memory for elements
//        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, WARP, NULL); // __local warpscan: encodes both number of used and status flags.
        oclCheckError(ciErr, CL_SUCCESS);
    }

    counter = 0;
    { // MemCopy Kernel
        kers.mem_cpy_ker = clCreateKernel(ctrl.cpProgram, is_sgm? "memcpy_wflags" : "memcpy_simple", &ciErr);
        oclCheckError(ciErr, CL_SUCCESS);

        ciErr |= clSetKernelArg(kers.mem_cpy_ker, counter++, sizeof(uint32_t), (void*)&buffs.N);
        if(is_sgm)
            ciErr |= clSetKernelArg(kers.mem_cpy_ker, counter++, sizeof(cl_mem), (void*)&buffs.gpu_flg);
        ciErr |= clSetKernelArg(kers.mem_cpy_ker, counter++, sizeof(cl_mem),   (void*)&buffs.gpu_inp);
        ciErr |= clSetKernelArg(kers.mem_cpy_ker, counter++, sizeof(cl_mem),   (void*)&buffs.gpu_out);
    }
}

void gpuToCpuTransfer(const uint32_t N, ElTp* cpu_out) {
    cl_int  ciErr;
    shrLog(stdlog, "GPU-to-CPU Transfer ...\n");
    ciErr = clEnqueueReadBuffer (
                        ctrl.cqCommandQueue, buffs.gpu_out, CL_TRUE,
                        0, N*sizeof(ElTp), cpu_out, 0, NULL, NULL
                );
    oclCheckError(ciErr, CL_SUCCESS);
}

void oclReleaseBuffKers(bool is_sgm) {
    shrLog(stdlog, "Releasing Kernels...\n");
    clReleaseKernel(kers.single_scan_ker);

    shrLog(stdlog, "Releasing GPU buffers ...\n");
    clReleaseMemObject(buffs.gpu_inp);
    if(is_sgm) clReleaseMemObject(buffs.gpu_flg);
    clReleaseMemObject(buffs.gpu_out);
    clReleaseMemObject(buffs.global_id);
    clReleaseMemObject(buffs.aggregates);
    clReleaseMemObject(buffs.incprefix);
    clReleaseMemObject(buffs.statusflgs);
}

void oclControlCleanUp() {
    shrLog(stdlog, "Releasing GPU program ...\n");
    clReleaseProgram(ctrl.cpProgram);

    shrLog(stdlog, "Releasing Command Queue ...\n");
    clReleaseCommandQueue(ctrl.cqCommandQueue);

    shrLog(stdlog, "Releasing Devices ...\n");
    free(ctrl.cdDevices);
        
    shrLog(stdlog, "Releasing GPU context ...\n");
    clReleaseContext(ctrl.cxGPUContext);
}
