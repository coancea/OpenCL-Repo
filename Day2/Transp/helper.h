#ifndef TRANSP_HELPER
#define TRANSP_HELPER

#define DEVICE_ID 1

typedef enum { 
    NAIVE_TRANSP,
    COALS_TRANSP,
    OPTIM_TRANSP
} TranspVers;

typedef enum {
    NAIVE_PROGRM,
    COALS_PROGRM,
    OPTIM_PROGRM
} ProgrmVers;

typedef struct OCLControl {
    cl_context          ctx;          // OpenCL context
    cl_device_id        device;       // OpenCL device
    cl_program          prog;         // OpenCL program
    cl_command_queue    queue;        // OpenCL command queue of the targe GPU device
} OclControl;

typedef struct OCLKernels {
    cl_kernel naiveTransp;  // naive transposition
    cl_kernel coalsTransp;  // coalesced transposition
    cl_kernel optimTransp;  // coalesced + chunked transposition
    cl_kernel naiveProgrm;  // uncoalesced accesses
    cl_kernel coalsProgrm;  // coalesced accesses by means of transposition
    cl_kernel optimProgrm;  // on the fly transposition
} OclKernels;

typedef struct OCLBuffers {
    // matrix shapes
    uint32_t height;
    uint32_t width;

    // input and result matrices (global memory)
    cl_mem  dA;     // heigth x width
    cl_mem  dB;     // height x width
    cl_mem  dAtr;   // width x height
    cl_mem  dBtr;   // width x height
} OclBuffers;

OclControl ctrl;
OclKernels kers;
OclBuffers buffs;

void initOclControl() {
    char    compile_opts[128];
    sprintf(compile_opts, "-D TILE=%d -D CHUNK=%d", TILE, CHUNK);
    opencl_init_command_queue(0, DEVICE_ID, &ctrl.device, &ctrl.ctx, &ctrl.queue);
    ctrl.prog = opencl_build_program(ctrl.ctx, ctrl.device, "kernels.cl", compile_opts);
}

void initOclBuffers ( const uint32_t height
                    , const uint32_t width
                    , real* hA
) {
    const size_t size = width * height * sizeof(real);
    cl_int error = CL_SUCCESS;
    buffs.dA = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, hA, &error);
    OPENCL_SUCCEED(error);

    buffs.dB = clCreateBuffer(ctrl.ctx, CL_MEM_WRITE_ONLY, size, NULL, &error);
    OPENCL_SUCCEED(error);

    buffs.dAtr = clCreateBuffer(ctrl.ctx, CL_MEM_WRITE_ONLY, size, NULL, &error);
    OPENCL_SUCCEED(error);

    buffs.dBtr = clCreateBuffer(ctrl.ctx, CL_MEM_WRITE_ONLY, size, NULL, &error);
    OPENCL_SUCCEED(error);

    buffs.height = height;
    buffs.width  = width;
}

void initTranspKernels() {
    cl_int error = CL_SUCCESS;
    
    { // naive
        kers.naiveTransp = clCreateKernel(ctrl.prog, "naiveTransp", &error);
        OPENCL_SUCCEED(error);
        clSetKernelArg(kers.naiveTransp, 0, sizeof(cl_mem), &buffs.dA);
        clSetKernelArg(kers.naiveTransp, 1, sizeof(cl_mem), &buffs.dB);
        clSetKernelArg(kers.naiveTransp, 2, sizeof(cl_int), &buffs.height);
        clSetKernelArg(kers.naiveTransp, 3, sizeof(cl_int), &buffs.width);
    }

    { // coalesced
        const size_t local_size = TILE * (TILE+1) * sizeof(real);
        kers.coalsTransp = clCreateKernel(ctrl.prog, "coalsTransp", &error);
        OPENCL_SUCCEED(error);
        clSetKernelArg(kers.coalsTransp, 0, sizeof(cl_mem), &buffs.dAtr);
        clSetKernelArg(kers.coalsTransp, 1, sizeof(cl_mem), &buffs.dBtr);
        clSetKernelArg(kers.coalsTransp, 2, sizeof(cl_int), &buffs.height);
        clSetKernelArg(kers.coalsTransp, 3, sizeof(cl_int), &buffs.width);
        clSetKernelArg(kers.coalsTransp, 4, local_size, NULL); // reserve space for local memory
    }

    { // coalesced + chunked
        const size_t local_size = CHUNK * TILE * (TILE+1) * sizeof(real);
        kers.optimTransp = clCreateKernel(ctrl.prog, "optimTransp", &error);
        OPENCL_SUCCEED(error);
        clSetKernelArg(kers.optimTransp, 0, sizeof(cl_mem), &buffs.dA);
        clSetKernelArg(kers.optimTransp, 1, sizeof(cl_mem), &buffs.dB);
        clSetKernelArg(kers.optimTransp, 2, sizeof(cl_int), &buffs.height);
        clSetKernelArg(kers.optimTransp, 3, sizeof(cl_int), &buffs.width);
        clSetKernelArg(kers.optimTransp, 4, local_size, NULL); // reserve space for local memory
    }
}

void initProgramKernels() {
    cl_int error = CL_SUCCESS;
    
    { // naive
        kers.naiveProgrm = clCreateKernel(ctrl.prog, "naiveProgrm", &error);
        OPENCL_SUCCEED(error);
        clSetKernelArg(kers.naiveProgrm, 0, sizeof(cl_mem), &buffs.dA);
        clSetKernelArg(kers.naiveProgrm, 1, sizeof(cl_mem), &buffs.dB);
        clSetKernelArg(kers.naiveProgrm, 2, sizeof(cl_int), &buffs.height);
        clSetKernelArg(kers.naiveProgrm, 3, sizeof(cl_int), &buffs.width);
    }

    { // coalesced
        kers.coalsProgrm = clCreateKernel(ctrl.prog, "coalsProgrm", &error);
        OPENCL_SUCCEED(error);
        clSetKernelArg(kers.coalsProgrm, 0, sizeof(cl_mem), &buffs.dAtr);
        clSetKernelArg(kers.coalsProgrm, 1, sizeof(cl_mem), &buffs.dBtr);
        clSetKernelArg(kers.coalsProgrm, 2, sizeof(cl_int), &buffs.height);
        clSetKernelArg(kers.coalsProgrm, 3, sizeof(cl_int), &buffs.width);
    }

    { // coalesced
        const size_t local_size = CHUNK * TILE * TILE * sizeof(real);
        kers.optimProgrm = clCreateKernel(ctrl.prog, "optimProgrm", &error);
        OPENCL_SUCCEED(error);
        clSetKernelArg(kers.optimProgrm, 0, sizeof(cl_mem), &buffs.dA);
        clSetKernelArg(kers.optimProgrm, 1, sizeof(cl_mem), &buffs.dB);
        clSetKernelArg(kers.optimProgrm, 2, sizeof(cl_int), &buffs.height);
        clSetKernelArg(kers.optimProgrm, 3, sizeof(cl_int), &buffs.width);
        clSetKernelArg(kers.optimProgrm, 4, local_size, NULL); // reserve space for local memory
    }
}

void gpuToCpuTransfer(const uint32_t N, real* cpu_out) {
    cl_int  ciErr;
    //fprintf(stderr, "GPU-to-CPU Transfer ...\n");
    ciErr = clEnqueueReadBuffer (
                        ctrl.queue, buffs.dB, CL_TRUE,
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
    clReleaseKernel(kers.naiveTransp);
    clReleaseKernel(kers.coalsTransp);
    clReleaseKernel(kers.optimTransp);
    clReleaseKernel(kers.naiveProgrm);
    clReleaseKernel(kers.coalsProgrm);
    clReleaseKernel(kers.optimProgrm);


    //fprintf(stderr, "Releasing GPU buffers ...\n");
    clReleaseMemObject(buffs.dA);
    clReleaseMemObject(buffs.dB);
    clReleaseMemObject(buffs.dAtr);
    clReleaseMemObject(buffs.dBtr);
}

#endif // TRANSP_HELPER
