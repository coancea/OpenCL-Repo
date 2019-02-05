#ifndef PARTITION2_H
#define PARTITION2_H

/***********************/
/*** Data Structures ***/
/***********************/

typedef struct OCLControl {
    cl_context          ctx;    // OpenCL context
    cl_device_id        device; // OpenCL device list
    cl_program          prog;   // OpenCL program
    cl_command_queue    queue;  // command queue of the targe GPU device
} OclControl;

typedef struct OCLKernels {
    cl_kernel grpRed;
    cl_kernel shortScan;
    cl_kernel grpScan;
} OclKernels;

typedef struct PartitionBUFFS {
    uint32_t      N;
    cl_mem        inp;  // [N]ElTp 
    cl_mem        tmpT; // [NUM_GROUPS_SCAN]int32_t
    cl_mem        tmpF; // [NUM_GROUPS_SCAN]int32_t
    cl_mem        out;  // [N]ElTp
} PartitionBuffs;

OclControl ctrl;
OclKernels kers;
PartitionBuffs buffs;

/********************/
/*** OpenCL stuff ***/
/********************/

void initOclControl() {
    char    compile_opts[128];
    sprintf(compile_opts, "-D lgWAVE=%d -D ELEMS_PER_THREAD=%d -D NE=%d -D ElTp=%s",
            lgWAVE, ELEMS_PER_THREAD, NE, ElTp_STR);
    
    //opencl_init_command_queue(0, GPU_DEV_ID, &ctrl.device, &ctrl.ctx, &ctrl.queue);
    opencl_init_command_queue_default(&ctrl.device, &ctrl.ctx, &ctrl.queue);
    ctrl.prog = opencl_build_program(ctrl.ctx, ctrl.device, "kernels.cl", compile_opts);
}


void initOclBuffers(const uint32_t N, ElTp* cpu_inp) {
    cl_int error = CL_SUCCESS;
    size_t size;

    // constants
    buffs.N = N;

    // global-memory input buffer
    size = N*sizeof(ElTp);
    buffs.inp = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, cpu_inp, &error);
    OPENCL_SUCCEED(error);

    // global-memory scan result
    size = N*sizeof(ElTp);
    buffs.out = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error);
    OPENCL_SUCCEED(error);

    // temporary value and flag buffers for the one-block scan
    size = NUM_GROUPS_SCAN * sizeof(int32_t);
    buffs.tmpT = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error);
    OPENCL_SUCCEED(error);

    size = NUM_GROUPS_SCAN * sizeof(int32_t);
    buffs.tmpF = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error);
    OPENCL_SUCCEED(error);
}

void initKernels() {
    cl_int error;

    kers.grpRed = clCreateKernel(ctrl.prog, "redPhaseKer", &error);
    OPENCL_SUCCEED(error);

    kers.shortScan = clCreateKernel(ctrl.prog, "shortScanKer", &error);
    OPENCL_SUCCEED(error);

    kers.grpScan = clCreateKernel(ctrl.prog, "scanPhaseKer", &error);
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
    clReleaseKernel(kers.grpRed);
    clReleaseKernel(kers.shortScan);
    clReleaseKernel(kers.grpScan);

    //fprintf(stderr, "Releasing GPU buffers ...\n");
    clReleaseMemObject(buffs.inp);
    clReleaseMemObject(buffs.out);
    clReleaseMemObject(buffs.tmpT);
    clReleaseMemObject(buffs.tmpF);
}

void freeOclControl() {
    //fprintf(stderr, "Releasing GPU program ...\n");
    clReleaseProgram(ctrl.prog);

    //fprintf(stderr, "Releasing Command Queue ...\n");
    clReleaseCommandQueue(ctrl.queue);
        
    //fprintf(stderr, "Releasing GPU context ...\n");
    clReleaseContext(ctrl.ctx);
}

/*********************************/
/*** various utility functions ***/
/*********************************/

inline ElTp spreadData(float r) { return (ElTp)(r * 10.0); }

void mkRandomDataset (const uint32_t N, ElTp* data) {
    for (uint32_t i = 0; i < N; ++i) {
        float r01 = rand() / (float)RAND_MAX;
        float r   = r01 - 0.5;
        data[i]   = spreadData(r);
    }
}

void validate(ElTp* A, ElTp* B, uint32_t sizeAB){
    for(uint32_t i = 0; i < sizeAB; i++)
      if (A[i] != B[i]) {
        printf("INVALID RESULT %d %d %d\n", i, A[i], B[i]);
        return;
      }
    printf("VALID RESULT!\n");
}

void cleanUpBuffer(size_t buf_len, cl_mem buf) {
    ElTp pattern = 0;
    cl_int error = 
        clEnqueueFillBuffer( ctrl.queue, buf, (void*)&pattern, sizeof(pattern),
                             0, buf_len*sizeof(pattern), 0, NULL, NULL );
    clFinish(ctrl.queue);
    OPENCL_SUCCEED(error);
}

/************************/
/*** Partition2 Fused ***/
/************************/

uint32_t getScanNumGroups(const uint32_t N) {
    uint32_t min_elem_per_group = WORKGROUP_SIZE*ELEMS_PER_THREAD;
    uint32_t num1 = (N + min_elem_per_group - 1) / min_elem_per_group;
    return (num1 < NUM_GROUPS_SCAN) ? num1 : NUM_GROUPS_SCAN;
}

inline size_t max_int(size_t a, size_t b) {
    return (a < b) ? b : a;
}

cl_int runPartition(PartitionBuffs arrs) {
    cl_int error = CL_SUCCESS;
    const uint32_t num_groups      = getScanNumGroups(arrs.N);
    const uint32_t elems_per_group0= (arrs.N + num_groups - 1) / num_groups;
    uint32_t min_elem_per_group = WORKGROUP_SIZE*ELEMS_PER_THREAD;
    const uint32_t elems_per_group = ((elems_per_group0 + min_elem_per_group - 1) / min_elem_per_group) * min_elem_per_group;
    //printf("N: %d, num_groups: %d, elems-per-group:%d\n", arrs.N, num_groups, elems_per_group);
    
    const size_t   localWorkSize  = WORKGROUP_SIZE;
    const size_t   globalWorkSize = WORKGROUP_SIZE * num_groups; // elems_per_group 

    {   // run intra-group reduction kernel
        const size_t local_size = 2 * WORKGROUP_SIZE * sizeof(int32_t);
        error |= clSetKernelArg(kers.grpRed, 0, sizeof(uint32_t), (void *)&arrs.N);
        error |= clSetKernelArg(kers.grpRed, 1, sizeof(uint32_t), (void *)&elems_per_group);
        error |= clSetKernelArg(kers.grpRed, 2, sizeof(cl_mem), (void*)&arrs.inp);
        error |= clSetKernelArg(kers.grpRed, 3, sizeof(cl_mem), (void*)&arrs.tmpT);
        error |= clSetKernelArg(kers.grpRed, 4, sizeof(cl_mem), (void*)&arrs.tmpF);
        error |= clSetKernelArg(kers.grpRed, 5, local_size, NULL);

        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.grpRed, 1, NULL,
                                        &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        //OPENCL_SUCCEED(error);
    }

    {   // run scan on the group resulted from the group-reduction kernel
		const size_t  localWorkSize = MAX_WORKGROUP_SIZE; //NUM_GROUPS_SCAN;
        const size_t  local_size = 2 * localWorkSize * sizeof(int32_t);
        error |= clSetKernelArg(kers.shortScan, 0, sizeof(uint32_t), (void *)&num_groups);
        error |= clSetKernelArg(kers.shortScan, 1, sizeof(cl_mem), (void*)&arrs.tmpT); // input
        error |= clSetKernelArg(kers.shortScan, 2, sizeof(cl_mem), (void*)&arrs.tmpF); // input
        error |= clSetKernelArg(kers.shortScan, 3, local_size, NULL);

        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.shortScan, 1, NULL,
                                        &localWorkSize, &localWorkSize, 0, NULL, NULL);
        //OPENCL_SUCCEED(error);
    }

    {   // run intra-block scan kernel while accumulating from the result of the previous step
        const size_t local_size = WORKGROUP_SIZE * ELEMS_PER_THREAD * sizeof(ElTp);
        error |= clSetKernelArg(kers.grpScan, 0, sizeof(uint32_t), (void *)&arrs.N);
        error |= clSetKernelArg(kers.grpScan, 1, sizeof(uint32_t), (void *)&elems_per_group);
        error |= clSetKernelArg(kers.grpScan, 2, sizeof(cl_mem), (void*)&arrs.inp);
        error |= clSetKernelArg(kers.grpScan, 3, sizeof(cl_mem), (void*)&arrs.tmpT);
        error |= clSetKernelArg(kers.grpScan, 4, sizeof(cl_mem), (void*)&arrs.tmpF);
        error |= clSetKernelArg(kers.grpScan, 5, sizeof(cl_mem), (void*)&arrs.out);
        error |= clSetKernelArg(kers.grpScan, 6, local_size, NULL);

        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.grpScan, 1, NULL,
                                        &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        //OPENCL_SUCCEED(error);
    }
    return error;
}

void profilePartition(PartitionBuffs arrs, ElTp* ref_arr, ElTp* res_arr) {
    int64_t elapsed, aft, bef;
    cl_int error = CL_SUCCESS;

    // dry run
    error |= runPartition(arrs);
    clFinish(ctrl.queue);
    OPENCL_SUCCEED(error);

    // measure timing
    bef = get_wall_time();
    for (int32_t i = 0; i < RUNS_GPU; i++) {
        error |= runPartition(arrs);
    }
    clFinish(ctrl.queue);
    aft = get_wall_time();
    elapsed = aft - bef;
    OPENCL_SUCCEED(error);

    {
        double microsecPerTransp = ((double)elapsed)/RUNS_GPU; 
        double bytesaccessed = 2 * arrs.N * sizeof(ElTp); // one read + one write
        double gigaBytesPerSec = (bytesaccessed * 1.0e-3f) / microsecPerTransp;
        printf("GPU Fused-Partition2 runs in: %ld microseconds; N: %d; GBytes/sec: %.2f  ...",
                elapsed/RUNS_GPU, buffs.N, gigaBytesPerSec);
    }

    { // transfer result to CPU and validate
        gpuToCpuTransfer(arrs.N, arrs.out, res_arr);
        validate(ref_arr, res_arr, arrs.N);
        memset(res_arr, 0, arrs.N*sizeof(ElTp));
        cleanUpBuffer(arrs.N, arrs.out);
    }
}

void goldenPartition (const uint32_t N, ElTp* cpu_inp, ElTp* cpu_out) {
    ElTp* cpu_tmp = (ElTp*)malloc(N*sizeof(ElTp));
    int64_t elapsed, aft, bef = get_wall_time();
    for(int r=0; r < 1; r++) {
      uint32_t num_true = 0, num_false = 0;
      for(uint32_t i=0; i<N; i++) {
        ElTp el = cpu_inp[i];
        if( pred(el) == 1 ) {
            cpu_out[num_true ++] = el;
        } else {
            cpu_tmp[num_false++] = el;
        }
      }
      for(uint32_t k=0; k<num_false; k++) {
        cpu_out[num_true+k] = cpu_tmp[k];
      }
    }
    aft = get_wall_time();
    elapsed = aft - bef;
    free(cpu_tmp);

    {
        double microsecPerTransp = (double)elapsed; 
        double bytesaccessed = 2 * N * sizeof(ElTp); // one read + one write
        double gigaBytesPerSec = (bytesaccessed * 1.0e-3f) / microsecPerTransp;
        printf("Sequential Golden Partition2 runs in: %ld microseconds; N: %d; GBytes/sec: %.2f\n",
                elapsed, N, gigaBytesPerSec);
    }
}

#endif // PARTITION2_H
