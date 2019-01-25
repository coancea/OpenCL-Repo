#ifndef MMM_HELPER
#define MMM_HELPER

typedef struct OCLControl {
    cl_context          ctx;          // OpenCL context
    cl_device_id        device;       // OpenCL device
    cl_program          prog;         // OpenCL program
    cl_command_queue    queue;        // OpenCL command queue of the targe GPU device
} OclControl;

typedef struct OCLKernels {
    cl_kernel naiveMMM;  // naive kernel
    cl_kernel blockMMM;  // block-tiled kernel
    cl_kernel rgblkMMM;  // register + block tiled kernel
} OclKernels;

typedef struct OCLBuffers {
    // matrix shapes
    uint32_t heightA;
    uint32_t widthA;
    uint32_t widthB;

    // input and result matrices (global memory)
    cl_mem  dA; // heigthA x widthA
    cl_mem  dB; // widthA  x widthB
    cl_mem  dC; // heightA x widthB
} OclBuffers;

OclControl ctrl;
OclKernels kers;
OclBuffers buffs;

void initOclControl() {
    char    compile_opts[128]; 
    sprintf(compile_opts, "-D TILE=%d -D RT=%d -D real=%s", TILE, RT, REAL_STR);
    //opencl_init_command_queue(0, DEVICE_ID, &ctrl.device, &ctrl.ctx, &ctrl.queue);
    opencl_init_command_queue_default(&ctrl.device, &ctrl.ctx, &ctrl.queue);
    ctrl.prog = opencl_build_program(ctrl.ctx, ctrl.device, "mmm.cl", compile_opts);
}

void initOclBuffers ( const uint32_t heightA
                    , const uint32_t widthB
                    , const uint32_t widthA
                    , real* hA
                    , real* hB
) {
    cl_int error = CL_SUCCESS;
    size_t size;
    size = widthA * heightA * sizeof(real);
    buffs.dA = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, hA, &error);
    OPENCL_SUCCEED(error);

    size = widthB * widthA * sizeof(real);
    buffs.dB = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, hB, &error);
    OPENCL_SUCCEED(error);

    size = heightA * widthB * sizeof(real);
    buffs.dC = clCreateBuffer(ctrl.ctx, CL_MEM_WRITE_ONLY, size, NULL, &error);
    OPENCL_SUCCEED(error);

    buffs.heightA = heightA;
    buffs.widthA  = widthA;
    buffs.widthB  = widthB;
}

void initKernels() {
    cl_int error = CL_SUCCESS;
    
    // naive
    kers.naiveMMM = clCreateKernel(ctrl.prog, "naiveMMM", &error);
    OPENCL_SUCCEED(error);
    clSetKernelArg(kers.naiveMMM, 0, sizeof(cl_mem), &buffs.dA);
    clSetKernelArg(kers.naiveMMM, 1, sizeof(cl_mem), &buffs.dB);
    clSetKernelArg(kers.naiveMMM, 2, sizeof(cl_mem), &buffs.dC);
    clSetKernelArg(kers.naiveMMM, 3, sizeof(cl_int), &buffs.heightA);
    clSetKernelArg(kers.naiveMMM, 4, sizeof(cl_int), &buffs.widthB);
    clSetKernelArg(kers.naiveMMM, 5, sizeof(cl_int), &buffs.widthA);

    // blocked
    kers.blockMMM = clCreateKernel(ctrl.prog, "blockMMM", &error);
    OPENCL_SUCCEED(error);
    clSetKernelArg(kers.blockMMM, 0, sizeof(cl_mem), &buffs.dA);
    clSetKernelArg(kers.blockMMM, 1, sizeof(cl_mem), &buffs.dB);
    clSetKernelArg(kers.blockMMM, 2, sizeof(cl_mem), &buffs.dC);
    clSetKernelArg(kers.blockMMM, 3, sizeof(cl_int), &buffs.heightA);
    clSetKernelArg(kers.blockMMM, 4, sizeof(cl_int), &buffs.widthB);
    clSetKernelArg(kers.blockMMM, 5, sizeof(cl_int), &buffs.widthA);

    // register + block tiling
    kers.rgblkMMM = clCreateKernel(ctrl.prog, "rgblkMMM", &error);
    OPENCL_SUCCEED(error);
    clSetKernelArg(kers.rgblkMMM, 0, sizeof(cl_mem), &buffs.dA);
    clSetKernelArg(kers.rgblkMMM, 1, sizeof(cl_mem), &buffs.dB);
    clSetKernelArg(kers.rgblkMMM, 2, sizeof(cl_mem), &buffs.dC);
    clSetKernelArg(kers.rgblkMMM, 3, sizeof(cl_int), &buffs.heightA);
    clSetKernelArg(kers.rgblkMMM, 4, sizeof(cl_int), &buffs.widthB);
    clSetKernelArg(kers.rgblkMMM, 5, sizeof(cl_int), &buffs.widthA);
}

void gpuToCpuTransfer(const uint32_t N, real* cpu_out) {
    cl_int  ciErr;
    //fprintf(stderr, "GPU-to-CPU Transfer ...\n");
    ciErr = clEnqueueReadBuffer (
                        ctrl.queue, buffs.dC, CL_TRUE,
                        0, N*sizeof(real), cpu_out, 0, NULL, NULL
                );
    OPENCL_SUCCEED(ciErr);
}

void freeOclControl() {
    //fprintf(stderr, "Releasing GPU program ...\n");
    clReleaseProgram(ctrl.prog);

    //fprintf(stderr, "Releasing Command Queue ...\n");
    clReleaseCommandQueue(ctrl.queue);
        
    //fprintf(stderr, "Releasing GPU context ...\n");
    clReleaseContext(ctrl.ctx);
}

void freeOclBuffKers() {
    //fprintf(stderr, "Releasing Kernels...\n");
    clReleaseKernel(kers.naiveMMM);
    clReleaseKernel(kers.blockMMM);
    clReleaseKernel(kers.rgblkMMM);

    //fprintf(stderr, "Releasing GPU buffers ...\n");
    clReleaseMemObject(buffs.dA);
    clReleaseMemObject(buffs.dB);
    clReleaseMemObject(buffs.dC);
}

#endif // MMM_HELPER
