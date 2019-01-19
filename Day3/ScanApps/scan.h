#ifndef SCAN_H
#define SCAN_H

typedef struct IncScanBUFFS {
    uint32_t      N;
    cl_mem        inp;  // input array holding `N` elements 
    cl_mem        tmp;  // holds `NUM_GROUPS_SCAN` elements
    cl_mem        out;  // result array holding `N` elements
} IncScanBuffs;

typedef struct SgmScanBUFFS {
    uint32_t      N;
    cl_mem        inp;  // input array holding `N` elements
    cl_mem        flg;  // input flags holding `N` bytes
    cl_mem        tmp_val;  // holds `NUM_GROUPS_SCAN` elements
    cl_mem        tmp_flg;  // holds `NUM_GROUPS_SCAN` bytes
    cl_mem        out;  // result array holding `N` elements
} SgmScanBuffs;

uint32_t getScanNumGroups(const uint32_t N) {
    const uint32_t min_elem_per_group = WORKGROUP_SIZE * ELEMS_PER_THREAD;
    const uint32_t num1 = (N + min_elem_per_group - 1) / min_elem_per_group;
    return (num1 < NUM_GROUPS_SCAN) ? num1 : NUM_GROUPS_SCAN;
}

/**********************/
/*** Inclusive Scan ***/
/**********************/

cl_int runScan(IncScanBuffs arrs) {
    cl_int error = CL_SUCCESS;
    const uint32_t num_groups      = getScanNumGroups(arrs.N);
    const uint32_t elems_per_group0= (arrs.N + num_groups - 1) / num_groups;
    const uint32_t min_elem_per_group = WORKGROUP_SIZE * ELEMS_PER_THREAD;
    const uint32_t elems_per_group = ((elems_per_group0 + min_elem_per_group - 1) / min_elem_per_group) * min_elem_per_group;

    const size_t   local_length   = WORKGROUP_SIZE * ELEMS_PER_THREAD;
    const size_t   localWorkSize  = WORKGROUP_SIZE;
    const size_t   globalWorkSize = WORKGROUP_SIZE * num_groups; // elems_per_group 

    {   // run intra-group reduction kernel
        error |= clSetKernelArg(kers.grpRed, 0, sizeof(uint32_t), (void *)&arrs.N);
        error |= clSetKernelArg(kers.grpRed, 1, sizeof(uint32_t), (void *)&elems_per_group);
        error |= clSetKernelArg(kers.grpRed, 2, sizeof(cl_mem), (void*)&arrs.inp);
        error |= clSetKernelArg(kers.grpRed, 3, sizeof(cl_mem), (void*)&arrs.tmp);
        error |= clSetKernelArg(kers.grpRed, 4, local_length * sizeof(ElTp), NULL);

        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.grpRed, 1, NULL,
                                        &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    }

    {   // run scan on the group resulted from the group-reduction kernel
        const size_t  localWorkSize = NUM_GROUPS_SCAN;
        error |= clSetKernelArg(kers.shortScan, 0, sizeof(uint32_t), (void *)&num_groups);
        error |= clSetKernelArg(kers.shortScan, 1, sizeof(cl_mem), (void*)&arrs.tmp); // input
        error |= clSetKernelArg(kers.shortScan, 2, NUM_GROUPS_SCAN * sizeof(ElTp), NULL);

        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.shortScan, 1, NULL,
                                        &localWorkSize, &localWorkSize, 0, NULL, NULL);
    }

    {   // run intra-block scan kernel while accumulating from the result of the previous step
        error |= clSetKernelArg(kers.grpScan, 0, sizeof(uint32_t), (void *)&arrs.N);
        error |= clSetKernelArg(kers.grpScan, 1, sizeof(uint32_t), (void *)&elems_per_group);
        error |= clSetKernelArg(kers.grpScan, 2, sizeof(cl_mem), (void*)&arrs.inp);
        error |= clSetKernelArg(kers.grpScan, 3, sizeof(cl_mem), (void*)&arrs.tmp);
        error |= clSetKernelArg(kers.grpScan, 4, sizeof(cl_mem), (void*)&arrs.out);
        error |= clSetKernelArg(kers.grpScan, 5, local_length * sizeof(ElTp), NULL);

        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.grpScan, 1, NULL,
                                        &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    }
    return error;
}

void profileScan(IncScanBuffs arrs, ElTp* ref_arr, ElTp* res_arr) {
    int64_t elapsed, aft, bef;
    cl_int error = CL_SUCCESS;

    // dry run
    error |= runScan(arrs);
    clFinish(ctrl.queue);
    OPENCL_SUCCEED(error);

    // measure timing
    bef = get_wall_time();
    for (int32_t i = 0; i < RUNS_GPU; i++) {
        error |= runScan(arrs);
    }
    clFinish(ctrl.queue);
    aft = get_wall_time();
    elapsed = aft - bef;
    OPENCL_SUCCEED(error);

    {
        double microsecPerTransp = ((double)elapsed)/RUNS_GPU; 
        double bytesaccessed = 2 * arrs.N * sizeof(ElTp); // one read + one write
        double gigaBytesPerSec = (bytesaccessed * 1.0e-3f) / microsecPerTransp;
        printf("GPU Inclusive-Scan runs in: %ld microseconds; N: %d; GBytes/sec: %.2f  ...",
                elapsed/RUNS_GPU, buffs.N, gigaBytesPerSec);
    }

    { // transfer result to CPU and validate
        gpuToCpuTransfer(arrs.N, arrs.out, res_arr);
        validate(ref_arr, res_arr, arrs.N);
        memset(res_arr, 0, arrs.N*sizeof(ElTp));
    }
}

/**********************/
/*** Segmented Scan ***/
/**********************/

cl_int runSgmScan(SgmScanBuffs arrs) {
    cl_int error = CL_SUCCESS;
    const uint32_t num_groups      = getScanNumGroups(arrs.N);
    const uint32_t elems_per_group0= (arrs.N + num_groups - 1) / num_groups;
    const uint32_t min_elem_per_group = WORKGROUP_SIZE * ELEMS_PER_THREAD;
    const uint32_t elems_per_group = ((elems_per_group0 + min_elem_per_group - 1) / min_elem_per_group) * min_elem_per_group;

    //printf("Number of groups: %d, Number elements per group: %d\n\n", num_groups, elems_per_group);

    const size_t   local_length   = WORKGROUP_SIZE * ELEMS_PER_THREAD;
    const size_t   localWorkSize  = WORKGROUP_SIZE;
    const size_t   globalWorkSize = WORKGROUP_SIZE * num_groups; // elems_per_group 

    {   // run intra-group reduction kernel
        error |= clSetKernelArg(kers.grpSgmRed, 0, sizeof(uint32_t), (void *)&arrs.N);
        error |= clSetKernelArg(kers.grpSgmRed, 1, sizeof(uint32_t), (void *)&elems_per_group);
        error |= clSetKernelArg(kers.grpSgmRed, 2, sizeof(cl_mem), (void*)&arrs.flg);
        error |= clSetKernelArg(kers.grpSgmRed, 3, sizeof(cl_mem), (void*)&arrs.inp);
        error |= clSetKernelArg(kers.grpSgmRed, 4, sizeof(cl_mem), (void*)&arrs.tmp_flg);
        error |= clSetKernelArg(kers.grpSgmRed, 5, sizeof(cl_mem), (void*)&arrs.tmp_val);
        error |= clSetKernelArg(kers.grpSgmRed, 6, local_length * sizeof(ElTp), NULL);

        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.grpSgmRed, 1, NULL,
                                        &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    }

    {   // run scan on the group resulted from the group-reduction kernel
        const size_t  localWorkSize = NUM_GROUPS_SCAN;
        error |= clSetKernelArg(kers.shortSgmScan, 0, sizeof(uint32_t), (void *)&num_groups);
        error |= clSetKernelArg(kers.shortSgmScan, 1, sizeof(cl_mem), (void*)&arrs.tmp_flg); // input
        error |= clSetKernelArg(kers.shortSgmScan, 2, sizeof(cl_mem), (void*)&arrs.tmp_val); // input
        error |= clSetKernelArg(kers.shortSgmScan, 3, NUM_GROUPS_SCAN * sizeof(ElTp), NULL);
        error |= clSetKernelArg(kers.shortSgmScan, 4, NUM_GROUPS_SCAN * sizeof(uint32_t), NULL);

        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.shortSgmScan, 1, NULL,
                                        &localWorkSize, &localWorkSize, 0, NULL, NULL);
    }

    {   // run intra-block scan kernel while accumulating from the result of the previous step
        error |= clSetKernelArg(kers.grpSgmScan, 0, sizeof(uint32_t), (void *)&arrs.N);
        error |= clSetKernelArg(kers.grpSgmScan, 1, sizeof(uint32_t), (void *)&elems_per_group);
        error |= clSetKernelArg(kers.grpSgmScan, 2, sizeof(cl_mem), (void*)&arrs.flg);
        error |= clSetKernelArg(kers.grpSgmScan, 3, sizeof(cl_mem), (void*)&arrs.inp);
        error |= clSetKernelArg(kers.grpSgmScan, 4, sizeof(cl_mem), (void*)&arrs.tmp_flg);
        error |= clSetKernelArg(kers.grpSgmScan, 5, sizeof(cl_mem), (void*)&arrs.tmp_val);
        error |= clSetKernelArg(kers.grpSgmScan, 6, sizeof(cl_mem), (void*)&arrs.out);
        error |= clSetKernelArg(kers.grpSgmScan, 7, local_length * sizeof(ElTp), NULL);

        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.grpSgmScan, 1, NULL,
                                        &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    }
    return error;
}

void profileSgmScan(SgmScanBuffs arrs, ElTp* ref_arr, ElTp* res_arr) {
    int64_t elapsed, aft, bef;
    cl_int error = CL_SUCCESS;

    // dry run
    error |= runSgmScan(arrs);
    clFinish(ctrl.queue);
    OPENCL_SUCCEED(error);

    // measure timing
    bef = get_wall_time();
    for (int32_t i = 0; i < RUNS_GPU; i++) {
        error |= runSgmScan(arrs);
    }
    clFinish(ctrl.queue);
    aft = get_wall_time();
    elapsed = aft - bef;
    OPENCL_SUCCEED(error);

    {
        double microsecPerTransp = ((double)elapsed)/RUNS_GPU; 
        double bytesaccessed = 2 * arrs.N * sizeof(ElTp); // one read + one write
        double gigaBytesPerSec = (bytesaccessed * 1.0e-3f) / microsecPerTransp;
        printf("GPU Segmented-Scan runs in: %ld microseconds; N: %d; GBytes/sec: %.2f  ...",
                elapsed/RUNS_GPU, buffs.N, gigaBytesPerSec);
    }

    { // transfer result to CPU and validate
        gpuToCpuTransfer(arrs.N, arrs.out, res_arr);
        validate(ref_arr, res_arr, arrs.N);
        memset(res_arr, 0, arrs.N*sizeof(ElTp));
    }
}

void goldenScan (uint8_t is_sgm, const uint32_t N, ElTp* cpu_inp, uint8_t* cpu_flags, ElTp* cpu_out) {
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

    {
        double microsecPerTransp = (double)elapsed; 
        double bytesaccessed = 2 * N * sizeof(ElTp); // one read + one write
        double gigaBytesPerSec = (bytesaccessed * 1.0e-3f) / microsecPerTransp;
        printf("Sequential Golden Scan runs in: %ld microseconds; N: %d; GBytes/sec: %.2f\n",
                elapsed, N, gigaBytesPerSec);
    }
}

#endif // SCAN_H
