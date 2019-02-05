typedef int     int32_t;
typedef uint    uint32_t;
typedef uchar   uint8_t;

#define WAVE    (1<<lgWAVE)

inline uint32_t pred(int32_t k) {
    return (1 - (k & 1));
}

inline int2 binOp(const int2 a, const int2 b) {
    int2 res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    return res;
}

/*****************************************/
/*** Inclusive Scan Helpers and Kernel ***/ 
/*****************************************/

inline int2
incScanWave(__local volatile int32_t* sh_data_x,
            __local volatile int32_t* sh_data_y,
            const size_t th_id
) { 
    const size_t lane = th_id & (WAVE-1);
    #pragma unroll
    for(uint32_t i=0; i<lgWAVE; i++) {
        const uint32_t p = (1<<i);
        if( lane >= p ) {
            int2 prev; prev.x = sh_data_x[th_id-p]; prev.y = sh_data_y[th_id-p];
            int2 curr; curr.x = sh_data_x[th_id];   curr.y = sh_data_y[th_id];
            int2 res = binOp(prev, curr);
            sh_data_x[th_id] = res.x; sh_data_y[th_id] = res.y;
        }
    }
    int2 res; res.x = sh_data_x[th_id]; res.y = sh_data_y[th_id]; return res;
}

inline int2 
incScanGroup ( __local volatile int32_t*   sh_data_x,
               __local volatile int32_t*   sh_data_y,
               const size_t tid
) {
    const size_t lane   = tid & (WAVE-1);
    const size_t waveid = tid >> lgWAVE;

    // perform scan at wave level
    int2 res = incScanWave(sh_data_x, sh_data_y, tid);

	// optimize for when the workgroup-size is exactly one WAVE
	if(get_local_size(0) == WAVE) { return res; }

    barrier(CLK_LOCAL_MEM_FENCE);

    // if last thread in a wave, record it at the beginning of sh_data
    if ( lane == (WAVE-1) ) {
        sh_data_x[waveid] = res.x;
        sh_data_y[waveid] = res.y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // first wave scans the per wave results (again)
    if( waveid == 0 ) incScanWave(sh_data_x, sh_data_y, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    // accumulate results from previous step;
    if (waveid > 0) {
        int2 prev; 
        prev.x = sh_data_x[waveid-1];
        prev.y = sh_data_y[waveid-1];
        res = binOp( prev, res );
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sh_data_x[tid] = res.x;
    sh_data_y[tid] = res.y;
    barrier(CLK_LOCAL_MEM_FENCE);
    return res;
}

__kernel void redPhaseKer ( 
        uint32_t            N,
        uint32_t            elem_per_group,
        __global ElTp*      d_inp,        // read-only,   [N]
        __global int32_t*   d_outT,       // write-only,  [number of workgroups]
        __global int32_t*   d_outF,       // write-only,  [number of workgroups]
        volatile __local    int32_t* locmem  // local memory [2*group-size]
) {
    const uint32_t tid = get_local_id(0);
    const uint32_t group_offset = get_group_id(0) * elem_per_group;
    int2 acc = 0;

    for(uint32_t k = 0; k < elem_per_group; k += get_local_size(0)) {
        uint32_t gid = group_offset + k + tid;
        if ( gid < N ) { 
            ElTp elm = d_inp[gid];
            int32_t c = pred(elm);
            acc.x += c;
            acc.y += (1 - c);
        }
    }

    { // scan in local memory
        volatile __local int32_t* locmem_x = locmem;
        volatile __local int32_t* locmem_y = locmem + get_local_size(0);
        locmem_x[tid] = acc.x; locmem_y[tid] = acc.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        acc = incScanGroup(locmem_x, locmem_y, tid);
    }
    
    if(tid == get_local_size(0)-1) {
        // write to global memory
        d_outT[get_group_id(0)] = acc.x;
        d_outF[get_group_id(0)] = acc.y;
    }
}

__kernel void shortScanKer( 
        uint32_t          N,
        __global int32_t* arrT,           // read-only [N]
        __global int32_t* arrF,           // read-only [N]
        volatile __local int32_t* locmem  // local memory [2*group-size]
) {
    volatile __local int32_t* locmem_x = locmem;
    volatile __local int32_t* locmem_y = locmem + get_local_size(0);
    const uint32_t tid = get_local_id(0);
    int2  acc = 0;

    for(uint32_t k = 0; k < N; k += get_local_size(0)) {
        int2 cur = 0;
        uint32_t gid = tid + k;
        if (gid < N) {
            cur.x = arrT[gid];
            cur.y = arrF[gid];
        }
        locmem_x[tid] = cur.x;
        locmem_y[tid] = cur.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        cur = incScanGroup(locmem_x, locmem_y, tid);
        cur = binOp(acc, cur);
        if(gid < N) {
            arrT[gid] = cur.x;
            arrF[gid] = cur.y;
        }
        // update the accumulator
        cur.x = locmem_x[get_local_size(0)-1];
        cur.y = locmem_y[get_local_size(0)-1];
        acc = binOp(acc, cur);
		barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/**
 * Last Exercise in Scan Applications: Optimized Partition2
 */
__kernel void scanPhaseKer ( 
        uint32_t            N,
        uint32_t            elem_per_group,
        __global ElTp*      d_inp,      // read-only,   [N]
        __global int32_t*   d_accT,     // read-only,   [number of workgroups]
        __global int32_t*   d_accF,     // read-only,   [number of workgroups]
        __global ElTp*      d_out,      // write-only,  [N]
        volatile __local int32_t* locmem  // local memory [group-size * ELEMS_PER_THREAD];
) {
    // we are going to reuse locmem for semantically-different arrays;
    // here are some (alised) definitions of such arrays:
    // locmem_e is a local-memory array holding ElTp elements; it is used:
    //   1. to copy NUM_ELEMS_PER_THREAD from global to local to register memory;
    //   2. in the last stage to order the elements first into local mem 
    //      and thus optimize coalescing for the write-back to global memory.
    volatile __local ElTp* locmem_e = (volatile __local ElTp*)locmem;

    // locmem_x and locmem_y are used for computing the number of elements
    // that succeed and fail under the predicate, respectively (i.e., the scan)
    volatile __local int32_t* locmem_x = locmem;
    volatile __local int32_t* locmem_y = locmem_x + get_local_size(0);

    const uint32_t tid = get_local_id(0);
    const uint32_t group_offset = get_group_id(0) * elem_per_group;
    const uint32_t group_chunk  = ELEMS_PER_THREAD * get_local_size(0);

    // accum.x and accum.y are the count of elements that succeed and fail
    // under the predicate, respectively, up until the previous workgroup.
    int2  accum = 0;

    // gii is the total numer of elements that succeed under the predicate
    uint32_t gii;

    if(tid == 0) {
        if( get_group_id(0) > 0) {
            accum.x = d_accT[get_group_id(0)-1];
            accum.y = d_accF[get_group_id(0)-1];
        }
        locmem_x[0] = accum.x; locmem_x[1] = accum.y; locmem_x[2] = d_accT[get_num_groups(0)-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    accum.x = locmem_x[0]; accum.y = locmem_x[1]; gii = locmem_x[2];

    // virtualization inside a workgroup
    for(uint32_t k = 0; k < elem_per_group; k += group_chunk) {
        barrier(CLK_LOCAL_MEM_FENCE);

        // 1. read "ELEMS_PER_THREAD*get_group_size(0)" elements from
        //    global to local memory ("locmem_e") in coalesced fasion
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            // ToDo Step 1: please fill in the implementation here!
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // "chunk" denotes the elements to-be-processed by the current thread 
        ElTp  chunk[ELEMS_PER_THREAD];
        uint32_t lind = tid*ELEMS_PER_THREAD; 
        uint32_t gind0 = lind + group_offset + k;

        // "tf" holds the per-thread true/false count (of elements
        //    that succeed/fail under predicate),
        //    i.e., for "ELEMS_PER_THREAD" values
        int2 tf = 0;     
 
        // 2. store in register memory this thread's elements and
        //      sequentially reduce in "tf"; for simplicity, one can
        //      redundantly call "pred" on the element; 
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            if(gind0 + i < N) {
                // ToDo Step 2: 
                // a) real current element "el" from local memory ("locmem_e")
                // b) save it in register memory "chunk"
                // c) call "pred(el)" and update "tf"
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. publish in local memory the per-thread true/flase
        //      count and perform intra-group scan 
        locmem_x[tid] = tf.x;
        locmem_y[tid] = tf.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        incScanGroup(locmem_x, locmem_y, tid);
        
        // 4. read the workgroup count of true/false elements
        int32_t num_ts = locmem_x[get_local_size(0)-1];
        int32_t num_fs = locmem_y[get_local_size(0)-1];

        // 5. read in "tf" the true/false count of the previous thread
        tf = 0;
        if (tid > 0) { tf.x = locmem_x[tid-1]; tf.y = locmem_y[tid-1]; }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 6. place the "ELEMS_PER_THREAD" elements of the current
        //      thread from register ("chunk") to local memory in
        //      the partial order of the workgroup partitioned result!
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            if(gind0 + i < N) {
                // ToDo Step 6:
                // a) redundantly call "pred" on "chunk[i]" element
                // b) find out the proper index in the partitioned
                //      result (in local memory) and update tf.x and tf.y;
                //      if the element fails under predicate, remember
                //      to add "num_ts" to local-mem index (why?)
                // c) write "chunk[i]" in local memory ("locmem_e")
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 7. write to global memory from local memory in coalesced fashion
        //      the elements are already in "good" ordering in local memory;
        //      we only need to adjust it.
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            uint32_t lind = i*get_local_size(0) + tid;
            uint32_t gind = group_offset + k + lind;
            if (gind < N) {
                uint32_t gmem_index;
                if (lind < num_ts) {
                    // corresponds to an element that succeeds under predicate;
                    // the global index is obtained by adding "accum.x", i.e.,
                    //   displacement up to the current iteration in the outer loop
                    //   (including the displacement due to the other threads).
                    gmem_index = lind + accum.x;
                } else {
                    // corresponds to an element that succeeds under predicate;
                    // the global index is obtained by adding "accum.y" and "(gii-num_ts)"
                    // The latter terms is the adjustment on the successful elements:
                    //  we need to add "gii", i.e., the global/total number of successful
                    //  elements and to substract "num_ts" (the local number of successful
                    //  elements)
                    gmem_index = lind - num_ts + accum.y + gii;
                }
                d_out[gmem_index] = locmem_e[lind];
            }
        }

        // 8. update accum for the next iteration
        accum.x += num_ts; accum.y += num_fs;
    }
}

