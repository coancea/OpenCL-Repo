typedef int     int32_t;
typedef uint    uint32_t;
typedef uchar   uint8_t;

#define WAVE (1<<lgWAVE)

/*****************/
/*** Utilities ***/ 
/*****************/

typedef struct FlgTup {
    uint8_t flg;
    ElTp    val;
} FlgTuple;

inline ElTp binOp(ElTp v1, ElTp v2) {
    return (v1 + v2);
}

inline FlgTuple binOpFlg(FlgTuple t1, FlgTuple t2) {
    FlgTuple res;
    if(t2.flg == 0) res.val = binOp(t1.val, t2.val);
    else            res.val = t2.val;
    res.flg = t1.flg | t2.flg;
    return res;
}

inline uint32_t pred(int32_t k) {
    return (1 - (k & 1));
}

/***********************/
/*** MemCopy Kernels ***/ 
/***********************/

__kernel void memcpy_wflags ( 
        uint32_t            N,
        __global uint8_t*   d_flg,      // read-only,  [N]
        __global ElTp*      d_inp,      // read-only,  [N]
        __global ElTp*      d_out       // write-only, [N]
) {
    uint32_t gid = get_global_id(0);
    if(gid < N) {
        d_out[gid] = d_inp[gid] + d_flg[gid];
    }
}

__kernel void memcpy_simple ( 
        uint32_t            N,
        __global ElTp*      d_inp,      // read-only,  [N]
        __global ElTp*      d_out       // write-only, [N]
) {
    uint32_t gid = get_global_id(0);
    if(gid < N) {
        d_out[gid] = d_inp[gid];
    }
}


/*****************************************/
/*** Inclusive Scan Helpers and Kernel ***/ 
/*****************************************/

/**
 * Exercise 2: fill in the implementation
 * You already have defined and can use:
 *   WAVE (=64) and lgWAVE (=6),
 * hence your k = lgWAVE and n = WAVE!
 * Also: the loop "forall i = h to n-1"
 *  is implicit, i.e., executed by all threads,
 *  and should be replaced with a condition, but
 *     "if(th_id >= h) ... " is the wrong one. WHY?
 */ 
inline ElTp
incScanWave(__local volatile ElTp* sh_data, const size_t th_id) {
    // ToDo: fill in the implementation here; let the return as it is!
    return sh_data[th_id];
}

inline ElTp 
incScanGroup ( __local volatile ElTp*   sh_data, const size_t tid) {
    const size_t lane   = tid & (WAVE-1);
    const size_t waveid = tid >> lgWAVE;

    // perform scan at wave level
    ElTp res = incScanWave(sh_data, tid);

	// optimize for when the workgroup-size is exactly one WAVE
	if(get_local_size(0) == WAVE) { return res; }

    barrier(CLK_LOCAL_MEM_FENCE);

    // if last thread in a wave, record it at the beginning of sh_data
    if ( lane == (WAVE-1) ) { sh_data[ waveid ] = res; }
    barrier(CLK_LOCAL_MEM_FENCE);

    // first wave scans the per wave results (again)
    if( waveid == 0 ) incScanWave(sh_data, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    // accumulate results from previous step;
    if (waveid > 0) {
        res = binOp( sh_data[waveid-1], res );
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sh_data[tid] = res;
    barrier(CLK_LOCAL_MEM_FENCE);
    return res;
}

__kernel void redPhaseKer ( 
        uint32_t            N,
        uint32_t            elem_per_group,
        __global ElTp*      d_inp,      // read-only,   [N]
        __global ElTp*      d_out,      // write-only,  [number of workgroups]
        volatile __local  ElTp* locmem  // local memory [group-size * ELEMS_PER_THREAD]
) {
    ElTp res = NE;
    const uint32_t tid = get_local_id(0);
    const uint32_t group_offset = get_group_id(0) * elem_per_group;
    const uint32_t group_chunk  = ELEMS_PER_THREAD * get_local_size(0);

    for(uint32_t k = 0; k < elem_per_group; k += group_chunk) {
        // 1. read ELEMS_PER_THREAD elements from global to local memory
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            uint32_t lind = i*get_local_size(0) + tid;
            uint32_t gind = group_offset + k + lind;
            ElTp v;
            if (gind < N) { v = d_inp[gind]; } else { v = NE; }
            locmem[lind] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 2. sequentially reduce from local memory ELEMS_PER_THREAD elements
        ElTp acc = NE;
        uint32_t loc_offset = tid * ELEMS_PER_THREAD;
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            acc = binOp(acc, locmem[loc_offset+i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. publish in local memory and perform intra-group scan 
        locmem[tid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        acc = incScanGroup(locmem, tid);
        if (tid == get_local_size(0)-1) {
            res = binOp(res, acc);
        }
    }
    
    if (tid == get_local_size(0)-1) {
        d_out[get_group_id(0)] = res;
    }
}

__kernel void shortScanKer( 
        uint32_t            N,
        __global ElTp*      arr,      // read-only,   [N]
        volatile __local  ElTp* locmem  // local memory [group-size]
) {
	const uint32_t group_size = get_local_size(0);
    const uint32_t tid = get_local_id(0);
    uint32_t gid = tid;
    ElTp acc = NE;
	for(uint32_t k = 0; k < N; k += group_size, gid += group_size) {
		ElTp cur = NE;
		if (gid < N) {
			cur = arr[gid];
		}
		locmem[tid] = cur;
		barrier(CLK_LOCAL_MEM_FENCE);
		cur = incScanGroup(locmem, tid);
		if(gid < N) {
			arr[gid] = binOp(acc, cur);
		}
		cur = locmem[group_size-1];
		acc = binOp(acc, cur);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void scanPhaseKer ( 
        uint32_t            N,
        uint32_t            elem_per_group,
        __global ElTp*      d_inp,      // read-only,   [N]
        __global ElTp*      d_acc,      // read-only,   [number of workgroups]
        __global ElTp*      d_out,      // write-only,  [N]
        volatile __local  ElTp* locmem  // local memory [group-size * ELEMS_PER_THREAD]
) {
    ElTp  accum;
    ElTp  chunk[ELEMS_PER_THREAD];
    const uint32_t tid = get_local_id(0);
    const uint32_t group_offset = get_group_id(0) * elem_per_group;
    const uint32_t group_chunk  = ELEMS_PER_THREAD * get_local_size(0);

    if(tid == 0){
        ElTp v = NE;
        if( get_group_id(0) > 0) {
            v = d_acc[get_group_id(0)-1];
        }
        locmem[0] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    accum = locmem[0];

    for(uint32_t k = 0; k < elem_per_group; k += group_chunk) {
        barrier(CLK_LOCAL_MEM_FENCE);

        // 1. read ELEMS_PER_THREAD elements from global to local memory
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            uint32_t lind = i*get_local_size(0) + tid;
            uint32_t gind = group_offset + k + lind;
            ElTp v;
            if (gind < N) { v = d_inp[gind]; } else { v = NE; }
            locmem[lind] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        ElTp tmp = NE;
        // 2. sequentially scan from local memory ELEMS_PER_THREAD elements
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            tmp = binOp(tmp, locmem[tid*ELEMS_PER_THREAD + i]);
            chunk[i] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. publish in local memory and perform intra-group scan 
        locmem[tid] = tmp;
        barrier(CLK_LOCAL_MEM_FENCE);
        tmp = incScanGroup(locmem, tid);
        
        // 4. read the previous element and complete the scan in local memory
        tmp = NE;
        if (tid > 0) tmp = locmem[tid-1];
        tmp = binOp(accum, tmp);
        accum = binOp(accum, locmem[get_local_size(0)-1]);
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            locmem[tid*ELEMS_PER_THREAD + i] = binOp(tmp, chunk[i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 5. write back to global memory
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            uint32_t lind = i*get_local_size(0) + tid;
            uint32_t gind = group_offset + k + lind;
            if (gind < N) {
                d_out[gind] = locmem[lind];
            }
        }
    }
}

/*****************************************/
/*** Segmented Scan Helpers and Kernel ***/ 
/*****************************************/

inline FlgTuple
incSgmScanWave  ( __local volatile uint32_t* sh_flag
                , __local volatile ElTp*    sh_data
                , const size_t th_id
) {
    const size_t lane = th_id & (WAVE-1);
    #pragma unroll
    for(uint32_t i=0; i<lgWAVE; i++) {
        const uint32_t p = (1<<i);
        if( lane >= p ) {
            FlgTuple tup1; tup1.flg = (uint8_t)sh_flag[th_id-p]; tup1.val = sh_data[th_id-p];
            FlgTuple tup2; tup2.flg = (uint8_t)sh_flag[th_id  ]; tup2.val = sh_data[th_id  ];
            FlgTuple tup3 = binOpFlg( tup1, tup2 );
            sh_flag[th_id] = tup3.flg; sh_data[th_id] = tup3.val;
        }
    }
    FlgTuple res; res.flg = sh_flag[th_id]; res.val = sh_data[th_id]; return res;
}

inline FlgTuple 
incSgmScanGroup( __local volatile uint32_t* sh_flag
                , __local volatile ElTp*    sh_data
                , const size_t tid
) {
    const size_t lane   = tid & (WAVE-1);
    const size_t waveid = tid >> lgWAVE;

    // perform scan at wave level
    FlgTuple res = incSgmScanWave(sh_flag, sh_data, tid);

	// optimize for when the workgroup-size is exactly one WAVE
	if(get_local_size(0) == WAVE) { return res; }

    barrier(CLK_LOCAL_MEM_FENCE);

    // if last thread in a wave, record it at the beginning of sh_data
    if ( lane == (WAVE-1) ) { sh_flag[waveid] = res.flg; sh_data[waveid] = res.val; }
    barrier(CLK_LOCAL_MEM_FENCE);

    // first wave scans the per wave results (again)
    if( waveid == 0 ) incSgmScanWave(sh_flag, sh_data, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    // accumulate results from previous step;
    if (waveid > 0) {
        FlgTuple prev;
        prev.flg = (uint8_t)sh_flag[waveid-1];
        prev.val = sh_data[waveid-1];
        res = binOpFlg( prev, res );
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sh_flag[tid] = res.flg;
    sh_data[tid] = res.val;
    barrier(CLK_LOCAL_MEM_FENCE);
    return res;
}  

__kernel void redPhaseSgmKer ( 
        uint32_t            N,
        uint32_t            elem_per_group,
        __global uint8_t*   d_flg,      // read-only,   [N]
        __global ElTp*      d_inp,      // read-only,   [N]
        __global uint8_t*   d_out_flg,  // write-only,  [number of workgroups]
        __global ElTp*      d_out_val,  // write-only,  [number of workgroups]
        volatile __local  ElTp* locmem  // local memory [group-size * ELEMS_PER_THREAD]
) {
    const uint32_t tid = get_local_id(0);
    const uint32_t group_offset = get_group_id(0) * elem_per_group;
    const uint32_t group_chunk  = ELEMS_PER_THREAD * get_local_size(0);
    FlgTuple chunk[ELEMS_PER_THREAD];
    FlgTuple res; res.flg = 0; res.val = NE;

    for(uint32_t k = 0; k < elem_per_group; k += group_chunk) {
        {   // 1. read ELEMS_PER_THREAD elements from global to local to register memory
            #pragma unroll
            for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
                uint32_t lind = i*get_local_size(0) + tid;
                uint32_t gind = group_offset + k + lind;
                ElTp v;
                if (gind < N) { v = d_inp[gind]; } else { v = NE; }
                locmem[lind] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            uint32_t loc_offset = tid * ELEMS_PER_THREAD;
            #pragma unroll
            for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
                chunk[i].val = locmem[loc_offset+i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        {   // 2. read ELEMS_PER_THREAD flags from global to local to register memory
            volatile __local uint32_t* locmem_flg = (volatile __local uint32_t*)locmem;
            #pragma unroll
            for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
                uint32_t lind = i*get_local_size(0) + tid;
                uint32_t gind = group_offset + k + lind;
                uint8_t  v;
                if (gind < N) { v = d_flg[gind]; } else { v = 0; }
                locmem_flg[lind] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            uint32_t loc_offset = tid * ELEMS_PER_THREAD;
            #pragma unroll
            for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
                chunk[i].flg = (uint8_t)locmem_flg[loc_offset+i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }


        {   // 3. sequentially reduce from register memory ELEMS_PER_THREAD elements
            volatile __local uint32_t* locmem_flg = (volatile __local uint32_t*)(locmem + get_local_size(0));
            FlgTuple acc = chunk[0];
            #pragma unroll
            for (uint32_t i = 1; i < ELEMS_PER_THREAD; i++) {
                acc = binOpFlg(acc, chunk[i]);
            }

            // 4. publish in local memory and perform intra-group scan
            locmem[tid]     = acc.val;
            locmem_flg[tid] = acc.flg;
            barrier(CLK_LOCAL_MEM_FENCE);
            acc = incSgmScanGroup(locmem_flg, locmem, tid);
            if (tid == get_local_size(0)-1) {
                res = binOpFlg(res, acc);
            }
        }
    }
    
    if (tid == get_local_size(0)-1) {
        d_out_flg[get_group_id(0)] = res.flg;
        d_out_val[get_group_id(0)] = res.val;
    }
}

__kernel void shortSgmScanKer( 
        uint32_t            N,
        __global uint8_t*   arr_flg,    // read-only,   [N]
        __global ElTp*      arr_val,    // read-only,   [N]
        volatile __local  ElTp*    locmem_val,  // local memory [group-size]
        volatile __local  uint32_t* locmem_flg   // local memory [group-size]
) {
	const uint32_t group_size = get_local_size(0);
    const uint32_t tid = get_local_id(0);
    uint32_t gid = tid;
    FlgTuple acc; acc.flg = 0; acc.val = NE;

	for(uint32_t k = 0; k < N; k += group_size, gid += group_size) {
		FlgTuple cur; cur.flg = 0; cur.val = NE;
		if (gid < N) {
			cur.flg = arr_flg[gid];
			cur.val = arr_val[gid];
		}
		locmem_flg[tid] = cur.flg;
		locmem_val[tid] = cur.val;
		barrier(CLK_LOCAL_MEM_FENCE);
		cur = incSgmScanGroup(locmem_flg, locmem_val, tid);
		if(gid < N) {
			cur = binOpFlg(acc, cur);
			arr_flg[gid] = cur.flg;
			arr_val[gid] = cur.val;
		}
		cur.flg = locmem_flg[group_size-1];
		cur.val = locmem_val[group_size-1];
		acc = binOpFlg(acc, cur);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void scanPhaseSgmKer ( 
        uint32_t            N,
        uint32_t            elem_per_group,
        __global uint8_t*   d_flg,      // read-only,   [N]
        __global ElTp*      d_inp,      // read-only,   [N]
        __global uint8_t*   d_acc_flg,  // read-only,   [number of workgroups]
        __global ElTp*      d_acc_val,  // read-only,   [number of workgroups]
        __global ElTp*      d_out,      // write-only,  [N]
        volatile __local  ElTp* locmem  // local memory [group-size * ELEMS_PER_THREAD]
) {
    FlgTuple  accum;
    FlgTuple  chunk[ELEMS_PER_THREAD];
    const uint32_t tid = get_local_id(0);
    const uint32_t group_offset = get_group_id(0) * elem_per_group;
    const uint32_t group_chunk  = ELEMS_PER_THREAD * get_local_size(0);

    if(tid == 0){
        accum.val = NE; accum.flg = 0;
        if( get_group_id(0) > 0) {
            accum.flg = d_acc_flg[get_group_id(0)-1];
            accum.val = d_acc_val[get_group_id(0)-1];
        }
        locmem[0] = accum.val; locmem[1] = accum.flg;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    accum.val = locmem[0]; accum.flg = (uint8_t)locmem[1];

    for(uint32_t k = 0; k < elem_per_group; k += group_chunk) {
        barrier(CLK_LOCAL_MEM_FENCE);

        {   // 1. read ELEMS_PER_THREAD elements from global to local to register memory
            #pragma unroll
            for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
                uint32_t lind = i*get_local_size(0) + tid;
                uint32_t gind = group_offset + k + lind;
                ElTp v;
                if ( (gind < N) && (lind+k < elem_per_group) ) { v = d_inp[gind]; } else { v = NE; }
                locmem[lind] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            uint32_t loc_offset = tid * ELEMS_PER_THREAD;
            #pragma unroll
            for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
                chunk[i].val = locmem[loc_offset+i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        {   // 2. read ELEMS_PER_THREAD flags from global to local to register memory
            volatile __local uint32_t* locmem_flg = (volatile __local uint32_t*)locmem;
            #pragma unroll
            for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
                uint32_t lind = i*get_local_size(0) + tid;
                uint32_t gind = group_offset + k + lind;
                uint8_t  v;
                if ( (gind < N) && (lind+k < elem_per_group) ) { v = d_flg[gind]; } else { v = 0; }
                locmem_flg[lind] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            uint32_t loc_offset = tid * ELEMS_PER_THREAD;
            #pragma unroll
            for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
                chunk[i].flg = (uint8_t)locmem_flg[loc_offset+i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        {   // 3. sequentially scan from register memory ELEMS_PER_THREAD elements
            FlgTuple tmp = chunk[0];
            #pragma unroll
            for (uint32_t i = 1; i < ELEMS_PER_THREAD; i++) {
                tmp = binOpFlg(tmp, chunk[i]);
                chunk[i] = tmp;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // 4. publish in local memory and perform intra-group scan
            volatile __local uint32_t* locmem_flg = (volatile __local uint32_t*)(locmem + get_local_size(0));
            locmem[tid]     = tmp.val;
            locmem_flg[tid] = tmp.flg;
            barrier(CLK_LOCAL_MEM_FENCE);
            tmp = incSgmScanGroup(locmem_flg, locmem, tid);
            
            // 5. read the previous element and complete the scan in local memory
            tmp.val = NE; tmp.flg = 0;
            if (tid > 0) { tmp.val = locmem[tid-1]; tmp.flg = locmem_flg[tid-1]; }
            tmp = binOpFlg(accum, tmp);
            #pragma unroll
            for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
                chunk[i] = binOpFlg(tmp, chunk[i]);
            }

            // 6. update accum for the next big iteration
            tmp.val = locmem[get_local_size(0)-1]; 
            tmp.flg = locmem_flg[get_local_size(0)-1];
            accum = binOpFlg(accum, tmp);
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        {  // 5. write back values from register to local to global memory
            #pragma unroll
            for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
                locmem[tid*ELEMS_PER_THREAD + i] = chunk[i].val;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll
            for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
                uint32_t lind = i*get_local_size(0) + tid;
                uint32_t gind = group_offset + k + lind;
                if( (gind < N) && (lind+k < elem_per_group) ) {
                    d_out[gind] = locmem[lind];
                }
            }
        }
    }
}

/*************************/ 
/*** Partition Kernels ***/ 
/*************************/
/**
 * Exercise 3: compute "tfs" and "ffs" from slides
 * "even" is implemented in function named "pred";
 * you may call it!
 * Replace the dummy implementation!
 */
__kernel void mapPredPartKer(  
        uint32_t            N,
        __global ElTp*      inp,    // read-only,    [N]
        __global uint32_t*  tfs,    // write-only,   [N]
        __global uint32_t*  ffs     // write-only,   [N]
) {
    uint32_t gid = get_global_id(0);
    if(gid < N) {
        uint32_t dummy1 = 1, dummy2 = 0;
        tfs[gid] = dummy1;
        ffs[gid] = dummy2;
    }
}

/**
 * Exercise 3: compute the scatter part.
 * Note that "isF" is actually "tmps" in the
 * slides, i.e., you have to add "i" to get "isF"!
 * It is OK to call "pred" again on "inp" to find in which case you are.
 * Try to minimize divergence: perform the global-memory
 * write outside the if-then-else!
 */
__kernel void scatterPartKer( 
        uint32_t            N,
        __global ElTp*      inp,    // read-only,    [N]
        __global uint32_t*  isT,    // read-only,    [N]
        __global uint32_t*  isF,    // read-only,    [N]
        __global ElTp*      out     // write-only,   [N]
) {
}

/***************************/ 
/*** SpMatVecMul Kernels ***/ 
/***************************/

__kernel void iniFlagsSpMVM(  
        uint32_t            N,
        __global uint8_t*   out
) {
    uint32_t gid = get_global_id(0);
    if(gid < N) {
        out[gid] = 0;
    }
}

/**
 * Exercise 4: computation of flag array,
 *             please implement!
 */ 
__kernel void mkFlagsSpMVM(  
        uint32_t            num_rows,
        __global uint32_t*  shape_scn,
        __global uint8_t*   flags
) {
    uint32_t gid = get_global_id(0);
    if(gid < num_rows) {
        flags[gid] = 1;
    }
}

/**
 * Exercise 4: computation of "prods",
 *             please implement!
 */ 
__kernel void mulPhaseSpMVM(  
        uint32_t            N,
        __global uint32_t*  mat_ind,
        __global ElTp*      mat_val,
        __global ElTp*      vect,
        __global ElTp*      out
) {
}

/**
 * Exercise 4: computation of final result,
 *             please implement!
 */ 
__kernel void getLastSpMVM(  
        uint32_t            num_rows,
        __global uint32_t*  shape_scn,
        __global ElTp*      sgm_mat,
        __global ElTp*      out
) {
}
