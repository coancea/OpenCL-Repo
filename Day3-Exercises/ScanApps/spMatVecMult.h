#ifndef SpMatVecMul_H
#define SpMatVecMul_H

/**
 *  let spMatVctMult [num_elms] [vct_len] [num_rows] 
 *                   (shp : [num_rows]i32)
 *                   (mat : [num_elms](i32,f32))
 *                   (vct : [vct_len]f32) : [num_rows]f32 =
 *      let shpscn    = scan (+) 0 shp          // can inclusive scan
 *      let flags     = replicate num_elms 0    // init flags kernel (1)
 *      let shp_inds  = map (\i -> if i > 0 then shpscn[i-1] else 0)
 *                          (iota num_rows)                            // build-flags kernel (2)
 *      let flags     = scatter flags shp_inds (replicate num_rows 1)  // build-flags kernel (2)
 *      let tmpmuls   = map (\(i,v) -> unsafe (v * vct[i]) ) mat       // multiplication kernel (3)
 *      let sgmmat    = segmented_scan (+) 0f32 flags tmpmuls          // call segmented scan
 *      in  map (\i -> sgmmat[i-1]) shp_sc                             // select kernel (4)
 */

typedef struct SpMVMBUFFS {
    uint32_t      num_elems;
    uint32_t      num_rows;
    uint32_t      vct_len;
    cl_mem        shape;    // [num_rows]uint32_t
    cl_mem        matval;   // [num_elems]ElTp
    cl_mem        matind;   // [num_elems]uint32_t
    cl_mem        vect;     // [vct_len]ElTp
    cl_mem        shpscn;   // [num_rows]uint32_t
    cl_mem        flags;    // [num_elems]uint8_t
    cl_mem        sgmmat;   // [num_elems]ElTp
    cl_mem        tmp_val;  // [`NUM_GROUPS_SCAN*ELEMS_PER_THREAD`] for sgmscan
    cl_mem        tmp_flg;  // [`NUM_GROUPS_SCAN*ELEMS_PER_THREAD`] for sgmscan
    cl_mem        out;      // [num_rows]ElTp 
} SpMVMBuffs;

cl_int runSpMatVectMul(SpMVMBuffs arrs) {
    cl_int error = CL_SUCCESS;
    const size_t localWorkSize  = WORKGROUP_SIZE;
    const size_t globalWorkSize0 = ((arrs.num_rows  + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) * WORKGROUP_SIZE;
    const size_t globalWorkSize1 = ((arrs.num_elems + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) * WORKGROUP_SIZE;
    
    { // scan the shapes; result is stored in `arrs.shpscn`
        IncScanBuffs scan_arrs;
        scan_arrs.N   = arrs.num_rows;
        scan_arrs.inp = arrs.shape;
        scan_arrs.tmp = arrs.tmp_val;
        scan_arrs.out = arrs.shpscn;
        error |= runScan(scan_arrs);
        OPENCL_SUCCEED(error);
    }

    { // call the kernel that initialize the flags array with zeroes (1)
        error |= clSetKernelArg(kers.iniFlagsSpMVM, 0, sizeof(uint32_t), (void *)&arrs.num_elems);
        error |= clSetKernelArg(kers.iniFlagsSpMVM, 1, sizeof(cl_mem), (void*)&arrs.flags);
        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.iniFlagsSpMVM, 1, NULL,
                                        &globalWorkSize1, &localWorkSize, 0, NULL, NULL);
        OPENCL_SUCCEED(error);
    }

    { // call the kernel that builds the flags (2)
        error |= clSetKernelArg(kers.mkFlagsSpMVM, 0, sizeof(uint32_t), (void *)&arrs.num_rows);
        error |= clSetKernelArg(kers.mkFlagsSpMVM, 1, sizeof(cl_mem), (void*)&arrs.shpscn);
        error |= clSetKernelArg(kers.mkFlagsSpMVM, 2, sizeof(cl_mem), (void*)&arrs.flags);
        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.mkFlagsSpMVM, 1, NULL,
                                        &globalWorkSize0, &localWorkSize, 0, NULL, NULL);
        OPENCL_SUCCEED(error);
    }

    { // call the kernel performing the multiplication between each matrix element and vector element (3)
        error |= clSetKernelArg(kers.mulPhaseSpMVM, 0, sizeof(uint32_t), (void *)&arrs.num_elems);
        error |= clSetKernelArg(kers.mulPhaseSpMVM, 1, sizeof(cl_mem), (void*)&arrs.matind);
        error |= clSetKernelArg(kers.mulPhaseSpMVM, 2, sizeof(cl_mem), (void*)&arrs.matval);
        error |= clSetKernelArg(kers.mulPhaseSpMVM, 3, sizeof(cl_mem), (void*)&arrs.vect);
        error |= clSetKernelArg(kers.mulPhaseSpMVM, 4, sizeof(cl_mem), (void*)&arrs.sgmmat);
        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.mulPhaseSpMVM, 1, NULL,
                                        &globalWorkSize1, &localWorkSize, 0, NULL, NULL);
        OPENCL_SUCCEED(error);
    }


    { // call the segmented scan kernel
        SgmScanBuffs scan_arrs;
        scan_arrs.N   = arrs.num_elems;
        scan_arrs.inp = arrs.sgmmat;
        scan_arrs.flg = arrs.flags;
        scan_arrs.tmp_val = arrs.tmp_val;
        scan_arrs.tmp_flg = arrs.tmp_flg;
        scan_arrs.out = arrs.sgmmat;
        error |= runSgmScan(scan_arrs);
        OPENCL_SUCCEED(error);
    }

    { // call the kernel that selects the last element on each row, i.e., the result (4)
        error |= clSetKernelArg(kers.getLastSpMVM, 0, sizeof(uint32_t), (void *)&arrs.num_rows);
        error |= clSetKernelArg(kers.getLastSpMVM, 1, sizeof(cl_mem), (void*)&arrs.shpscn);
        error |= clSetKernelArg(kers.getLastSpMVM, 2, sizeof(cl_mem), (void*)&arrs.sgmmat);
        error |= clSetKernelArg(kers.getLastSpMVM, 3, sizeof(cl_mem), (void*)&arrs.out);
        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.getLastSpMVM, 1, NULL,
                                        &globalWorkSize0, &localWorkSize, 0, NULL, NULL);
        OPENCL_SUCCEED(error);
    }

    return error;
}

void goldenSpMVM( uint32_t num_elems, uint32_t num_rows,
                  uint32_t* shape, uint32_t* matind, ElTp* matval,
                  ElTp* vect, ElTp* res
) {
    int64_t elapsed, aft, bef = get_wall_time();

    for(uint32_t r=0; r<1; r++) {
        uint32_t offset = 0;
        for(uint32_t i = 0; i < num_rows; i++) {
            uint32_t row_len = shape[i];
            ElTp accum = 0;
            for(uint32_t k=0; k < row_len; k++) {
                uint32_t ind  = matind[offset+k];
                ElTp     mel  = matval[offset+k];
                accum += mel * vect[ind];
            }
            res[i] = accum;
            offset += row_len;
        }
    }

    aft = get_wall_time();
    elapsed = aft - bef;
    
    {
        double microsecPerTransp = (double)elapsed; 
        double bytesaccessed = 3 * num_elems * sizeof(ElTp); // one read + one write
        double gigaBytesPerSec = (bytesaccessed * 1.0e-3f) / microsecPerTransp;
        printf("Sequential Golden SpMatVecMult runs in: %ld microseconds; N: %d; GBytes/sec: %.2f\n",
                elapsed, num_elems, gigaBytesPerSec);
    }
}

void makeDataset(uint32_t  N,
                 uint32_t  num_rows,
                 uint32_t* shape,  // [num_rows] 
                 uint32_t* matind, // [N]
                 ElTp*     vect    // [N]
) {
    // make shape
    const uint32_t num_cols = N / num_rows;
    for(uint32_t k = 0; k < num_rows-1; k++) {
        shape[k] = num_cols;
    }
    shape[num_rows-1] = N - (num_rows-1)*num_cols;

    // make vector with random values in (0,5)
    for (uint32_t i = 0; i < N; ++i) {
        float r01 = rand() / (float)RAND_MAX;
        int32_t r = (int) (r01 * 5.0);
        vect[i] = r;
    }

    // make matrix indices
    uint32_t offset = 0;
    for (uint32_t i = 0; i < num_rows; i++) {
        uint32_t row_len = shape[i];
        for (uint32_t j = 0; j < row_len; j++) {
            uint32_t colind = (num_rows-i)*j;
            if (colind >= N) colind = N - 1;
            matind[offset + j] = colind;
        }
        offset += row_len;
    }
}

void profileSpMatVectMul(SgmScanBuffs sgm_arrs, ElTp* cpu_matval) {
    int64_t elapsed, aft, bef;
    cl_int error = CL_SUCCESS;
    const uint32_t num_elems = sgm_arrs.N;
    const uint32_t num_rows  = 1000;
    uint32_t* cpu_matind = (uint32_t*)malloc(num_elems*sizeof(uint32_t));
    uint32_t* cpu_shape  = (uint32_t*)malloc(num_rows *sizeof(uint32_t));
    ElTp* cpu_vect = (ElTp*)malloc(num_elems*sizeof(ElTp));
    ElTp* cpu_ref  = (ElTp*)malloc(num_rows *sizeof(ElTp));
    ElTp* cpu_res  = (ElTp*)malloc(num_rows *sizeof(ElTp));
    SpMVMBuffs arrs;

    // fill in the cpu_shape, cpu_matind and cpu_vect arrays
    makeDataset(num_elems, num_rows, cpu_shape, cpu_matind, cpu_vect);

    // run sequential golden implementation to compute the reference result
    goldenSpMVM(num_elems, num_rows, cpu_shape, cpu_matind, cpu_matval, cpu_vect, cpu_ref);
    
    // allocate extra GPU buffers and initialize
    {
        size_t size;
        cl_int error = CL_SUCCESS;

        arrs.num_elems = num_elems;
        arrs.num_rows  = num_rows;
        arrs.vct_len   = num_elems;
        arrs.matval    = sgm_arrs.inp;
        arrs.flags     = sgm_arrs.flg;
        arrs.tmp_val   = sgm_arrs.tmp_val;
        arrs.tmp_flg   = sgm_arrs.tmp_flg;
        arrs.sgmmat    = sgm_arrs.out;
 
        size = num_rows * sizeof(uint32_t);
        arrs.shape = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, cpu_shape, &error);
        OPENCL_SUCCEED(error);

        size = num_elems * sizeof(uint32_t);
        arrs.matind = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, cpu_matind, &error);
        OPENCL_SUCCEED(error);
        
        size = num_elems * sizeof(uint32_t);
        arrs.vect = clCreateBuffer(ctrl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, cpu_vect, &error);
        OPENCL_SUCCEED(error);

        size = num_rows * sizeof(ElTp);
        arrs.shpscn = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error);
        OPENCL_SUCCEED(error);

        size = num_rows * sizeof(ElTp);
        arrs.out = clCreateBuffer(ctrl.ctx, CL_MEM_READ_WRITE, size, NULL, &error);
        OPENCL_SUCCEED(error);
    }

    // dry run
    error |= runSpMatVectMul(arrs);
    clFinish(ctrl.queue);
    OPENCL_SUCCEED(error);

    // measure timing
    bef = get_wall_time();
    for (int32_t i = 0; i < RUNS_GPU; i++) {
        error |= runSpMatVectMul(arrs);
    }
    clFinish(ctrl.queue);
    aft = get_wall_time();
    elapsed = aft - bef;
    OPENCL_SUCCEED(error);

    {
        double microsecPerTransp = ((double)elapsed)/RUNS_GPU; 
        double bytesaccessed = 3 * arrs.num_elems * sizeof(ElTp); // one read + one write
        double gigaBytesPerSec = (bytesaccessed * 1.0e-3f) / microsecPerTransp;
        printf("GPU SpMatVctMult runs in: %ld microseconds; N: %d; GBytes/sec: %.2f  ...",
                elapsed/RUNS_GPU, num_elems, gigaBytesPerSec);
    }

    { // transfer result to CPU and validate
        gpuToCpuTransfer(arrs.num_rows, arrs.out, cpu_res);
        validate(cpu_ref, cpu_res, arrs.num_rows);
    }

    { // free CPU and GPU buffers allocated in this function
        free(cpu_matind);
        free(cpu_shape);
        free(cpu_vect);
        free(cpu_ref);
        free(cpu_res);

        clReleaseMemObject(arrs.shape);
        clReleaseMemObject(arrs.matind);
        clReleaseMemObject(arrs.vect);
        clReleaseMemObject(arrs.shpscn);
        clReleaseMemObject(arrs.out);
    }
}

#endif // SpMatVecMul_H
