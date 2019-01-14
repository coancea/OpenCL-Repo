typedef struct OCLControl {
    cl_context          ctx;            // OpenCL context
    cl_device_id        device;      // OpenCL device list
    cl_program          prog;      // OpenCL program
    cl_command_queue    queue; // command queue of the targe GPU device
} OclControl;

typedef struct OCLKernels {
    cl_kernel single_scan_ker;  // Single-Pass Scan Kernel
    cl_kernel mem_cpy_ker;  // Memcopy Kernel
} OclKernels;

typedef struct OCLBuffers {
    // constants
    uint32_t N;

    // main input and result arrays (global memory)
    cl_mem  gpu_inp;
    cl_mem  gpu_flg;
    cl_mem  gpu_out;

    // For Single-Pass Scan: various temporary arrays
    cl_mem  global_id;
    cl_mem  aggregates; // includes incprefix
    cl_mem  incprefix;
    cl_mem  statusflgs; // includes the flags of aggregates and incprefix
} OclBuffers;

//CpuArrays  arrs;
OclControl ctrl;
OclKernels kers;
OclBuffers buffs;

int32_t getNumElemPerThread() {
    float num = (ELEMS_PER_THREAD * 4.0) / sizeof(ElTp);
    return (int32_t) num;
}

inline size_t getNumBlocks(const uint32_t N) {
    const size_t numelems_group = WORKGROUP_SIZE * getNumElemPerThread();
    return (N + numelems_group - 1) / numelems_group;
}

void initOclControl() {
    int32_t num_elem_per_thread     = getNumElemPerThread();
    int32_t sgm_num_elem_per_thread = getNumElemPerThread();
    char    compile_opts[128];
    sprintf(compile_opts, "-D lgWARP=%d -D ELEMS_PER_THREAD=%d -D SGM_ELEMS_PER_THREAD=%d", 
                          lgWARP, num_elem_per_thread, sgm_num_elem_per_thread);
    
    //opencl_init_command_queue(0, GPU_DEV_ID, &ctrl.device, &ctrl.ctx, &ctrl.queue);
    opencl_init_command_queue_default(&ctrl.device, &ctrl.ctx, &ctrl.queue);
    ctrl.prog = opencl_build_program(ctrl.ctx, ctrl.device, "SinglePassScanKer.cl", compile_opts);
}

void initOclBuffers(const uint32_t N, bool is_sgm, uint8_t* cpu_flg, ElTp* cpu_inp) {
    cl_int error = CL_SUCCESS;
    cl_int cpu_global_id[1] = {0};
    size_t size;

    // constants
    size_t num_blocks = getNumBlocks(N);
    buffs.N = N;

    // global-memory input buffer
    size = N*sizeof(ElTp);
    buffs.gpu_inp = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, cpu_inp, &error);
    OPENCL_SUCCEED(error);

    // global-memory flags buffer
    if (is_sgm) {
        size = N*sizeof(uint8_t);
        buffs.gpu_flg = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, cpu_flg, &error);
        OPENCL_SUCCEED(error);
    }
    // global-memory scan result
    size = N*sizeof(ElTp);
    buffs.gpu_out = clCreateBuffer(ctrl.ctx, CL_MEM_WRITE_ONLY, size, NULL, &error);
    OPENCL_SUCCEED(error);

    // global-memory intermediate-computation buffers
    {   // global_id
        buffs.global_id = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(int32_t), cpu_global_id, &error);
        OPENCL_SUCCEED(error);

        // allocate contiguous `aggregates` and `incprefix`
        size = num_blocks * sizeof(ElTp);
        buffs.aggregates = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error );
        OPENCL_SUCCEED(error);
        buffs.incprefix  = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error );
        OPENCL_SUCCEED(error);

        // allocate contiguous `status flags`, and the flags of `aggregate` and `incprefix`
        size = num_blocks * sizeof(uint8_t);
        buffs.statusflgs = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error ); 
        OPENCL_SUCCEED(error);
    }
}

void initKernels(const bool is_sgm) {
    cl_int ciErr;
    unsigned int counter = 0;

    { // Scan kernels
        int32_t num_elem_per_thread = getNumElemPerThread();
        const size_t LOCAL_SIZE_EXCG = WORKGROUP_SIZE * num_elem_per_thread;

        kers.single_scan_ker = clCreateKernel(ctrl.prog, is_sgm? "singlePassSgmScanKer" : "singlePassScanKer", &ciErr);
        OPENCL_SUCCEED(ciErr);

        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(uint32_t), (void *)&buffs.N);
        if (is_sgm) {
            ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void*)&buffs.gpu_flg); // global flags
        }
        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void*)&buffs.gpu_inp); // global input
        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void*)&buffs.gpu_out); // global output
        OPENCL_SUCCEED(ciErr);

        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void *)&buffs.global_id );
        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void *)&buffs.aggregates);
        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void *)&buffs.incprefix );
        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(cl_mem), (void *)&buffs.statusflgs);
        OPENCL_SUCCEED(ciErr);

        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, sizeof(int32_t), NULL); // __local block_id
        ciErr |= clSetKernelArg(kers.single_scan_ker, counter++, LOCAL_SIZE_EXCG * sizeof(ElTp), NULL); // __local exchange
        OPENCL_SUCCEED(ciErr);
    }

    counter = 0;
    { // MemCopy Kernel
        kers.mem_cpy_ker = clCreateKernel(ctrl.prog, is_sgm? "memcpy_wflags" : "memcpy_simple", &ciErr);
        OPENCL_SUCCEED(ciErr);

        ciErr |= clSetKernelArg(kers.mem_cpy_ker, counter++, sizeof(uint32_t), (void*)&buffs.N);
        if(is_sgm)
            ciErr |= clSetKernelArg(kers.mem_cpy_ker, counter++, sizeof(cl_mem), (void*)&buffs.gpu_flg);
        ciErr |= clSetKernelArg(kers.mem_cpy_ker, counter++, sizeof(cl_mem),   (void*)&buffs.gpu_inp);
        ciErr |= clSetKernelArg(kers.mem_cpy_ker, counter++, sizeof(cl_mem),   (void*)&buffs.gpu_out);
        OPENCL_SUCCEED(ciErr);
    }
}

void gpuToCpuTransfer(const uint32_t N, ElTp* cpu_out) {
    cl_int  ciErr;
    fprintf(stderr, "GPU-to-CPU Transfer ...\n");
    ciErr = clEnqueueReadBuffer (
                        ctrl.queue, buffs.gpu_out, CL_TRUE,
                        0, N*sizeof(ElTp), cpu_out, 0, NULL, NULL
                );
    OPENCL_SUCCEED(ciErr);
}

void freeOclBuffKers(bool is_sgm) {
    fprintf(stderr, "Releasing Kernels...\n");
    clReleaseKernel(kers.single_scan_ker);

    fprintf(stderr, "Releasing GPU buffers ...\n");
    clReleaseMemObject(buffs.gpu_inp);
    if(is_sgm) clReleaseMemObject(buffs.gpu_flg);
    clReleaseMemObject(buffs.gpu_out);
    clReleaseMemObject(buffs.global_id);
    clReleaseMemObject(buffs.aggregates);
    clReleaseMemObject(buffs.incprefix);
    clReleaseMemObject(buffs.statusflgs);
}

void freeOclControl() {
    fprintf(stderr, "Releasing GPU program ...\n");
    clReleaseProgram(ctrl.prog);

    fprintf(stderr, "Releasing Command Queue ...\n");
    clReleaseCommandQueue(ctrl.queue);
        
    fprintf(stderr, "Releasing GPU context ...\n");
    clReleaseContext(ctrl.ctx);
}
