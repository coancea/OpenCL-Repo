#ifndef PARTITION_H
#define PARTITION_H

/**
 * let partition2 [n] 't (p: (t -> bool)) (arr: [n]t) : ([n]t , i32) =
 *   let cs  = map p arr                            // mapPred kernel
 *   let tfs = map (\ f -> if f then 1 else 0) cs   // mapPred kernel
 *   let ffs = map (\ f -> if f then 0 else 1) cs   // mapPred kernel
 *   let isF = scan (+) 0 ffs                       // first scan
 *   let isT = scan (+) 0 tfs                       // second scan
 *   // (isT, isF) = unzip <| scan (\(a1,b1) (a2,b2) -> (a1+a2, b1+b2)) <| zip tfs ffs
 *   let i   = isT[n-1]                                 // scatter kernel
 *   let isF'= map (+i) isF                             // scatter kernel
 *   let inds= map (\c iT iF->if c then iT-1 else iF-1) // scatter kernel
 *                 (zip cs isT isF')                    
 *   let r = scatter (scratch n t) inds arr             // scatter kernel
 *   in (r, i)
 **/

typedef struct PartitionBUFFS {
    uint32_t      N;
    cl_mem        inp;  // [N]t 
    cl_mem        tfs;  // [N]uint32_t
    cl_mem        ffs;  // [N]uint32_t
    cl_mem        isT;  // [N]uint32_t
    cl_mem        isF;  // [N]uint32_t
    cl_mem        tmp; // [`NUM_GROUPS_SCAN*ELEMS_PER_THREAD`] elements
    cl_mem        out;  // [N]t 
} PartitionBuffs;

/**
 * Exercise 3: 
 */
cl_int runPartition(PartitionBuffs arrs) {
    cl_int error = CL_SUCCESS;
    const size_t localWorkSize  = WORKGROUP_SIZE;
    const size_t globalWorkSize = ((arrs.N + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) * WORKGROUP_SIZE; 
    
    { // call the kernel that maps the predicate and computes tfs and ffs
        error |= clSetKernelArg(kers.mapPredPart, 0, sizeof(uint32_t), (void *)&arrs.N);
        error |= clSetKernelArg(kers.mapPredPart, 1, sizeof(cl_mem), (void*)&arrs.inp);
        error |= clSetKernelArg(kers.mapPredPart, 2, sizeof(cl_mem), (void*)&arrs.tfs);
        error |= clSetKernelArg(kers.mapPredPart, 3, sizeof(cl_mem), (void*)&arrs.ffs);
        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.mapPredPart, 1, NULL,
                                        &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    }

    { // call the two scan kernels; first `isF` then `isT`
        IncScanBuffs scan_arrs;
        scan_arrs.N   = arrs.N;
        scan_arrs.inp = arrs.ffs;
        scan_arrs.tmp = arrs.tmp;
        scan_arrs.out = arrs.isF;
        error |= runScan(scan_arrs);

        scan_arrs.inp = arrs.tfs;
        scan_arrs.tmp = arrs.tmp;
        scan_arrs.out = arrs.isT;
        error |= runScan(scan_arrs);
    }

    { // call the scatter kernel: "isF" is actually "tmps" from slides
      // isT and isF are the result of scan
        error |= clSetKernelArg(kers.scatterPart, 0, sizeof(uint32_t), (void *)&arrs.N);
        error |= clSetKernelArg(kers.scatterPart, 1, sizeof(cl_mem), (void*)&arrs.inp);
        error |= clSetKernelArg(kers.scatterPart, 2, sizeof(cl_mem), (void*)&arrs.isT);
        error |= clSetKernelArg(kers.scatterPart, 3, sizeof(cl_mem), (void*)&arrs.isF);
        error |= clSetKernelArg(kers.scatterPart, 4, sizeof(cl_mem), (void*)&arrs.out);
        error |= clEnqueueNDRangeKernel(ctrl.queue, kers.scatterPart, 1, NULL,
                                        &globalWorkSize, &localWorkSize, 0, NULL, NULL);
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
        printf("GPU Partition2 Unfused runs in: %ld microseconds; N: %d; GBytes/sec: %.2f  ...",
                elapsed/RUNS_GPU, buffs.N, gigaBytesPerSec);
    }

    { // transfer result to CPU and validate
        uint32_t num_trues;
        cl_int  error = clEnqueueReadBuffer (
                            ctrl.queue, arrs.isT, CL_TRUE, (arrs.N-1)*sizeof(ElTp),
                            sizeof(ElTp), &num_trues, 0, NULL, NULL
                        );
        OPENCL_SUCCEED(error);
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

#endif //PARTITION_H
