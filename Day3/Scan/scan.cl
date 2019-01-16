typedef int     int32_t;
typedef uint    uint32_t;
typedef uchar   uint8_t;

#include "bridge.h" 

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

inline ElTp
incScanWarp(__local volatile ElTp* sh_data, const size_t th_id) { 
    const size_t lane = th_id & (WARP-1);
    #pragma unroll
    for(uint32_t i=0; i<lgWARP; i++) {
        const uint32_t p = (1<<i);
        if( lane >= p ) sh_data[th_id] = binOp( sh_data[th_id-p], sh_data[th_id] );
    }
    return sh_data[th_id];
}

inline ElTp 
incScanGroup ( __local volatile ElTp*   sh_data, const size_t tid) {
    const size_t lane   = tid & (WARP-1);
    const size_t warpid = tid >> lgWARP;

    // perform scan at warp level
    ElTp res = incScanWarp(sh_data, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    // if last thread in a warp, record it at the beginning of sh_data
    if ( lane == (WARP-1) ) { sh_data[ warpid ] = res; }
    barrier(CLK_LOCAL_MEM_FENCE);

    // first warp scans the per warp results (again)
    if( warpid == 0 ) incScanWarp(sh_data, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    // accumulate results from previous step;
    if (warpid > 0) {
        res = binOp( sh_data[warpid-1], res );
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
            if ( (gind < N) && (lind+k < elem_per_group) ) { v = d_inp[gind]; } else { v = NE; }
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
    const uint32_t tid = get_local_id(0);
    const uint32_t gid = get_global_id(0);
    ElTp acc = NE;
    if (gid < N) {
        acc = arr[gid];
    }
    locmem[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    acc = incScanGroup(locmem, tid);
    if(gid < N) {
        arr[gid] = acc;
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
            if ( (gind < N) && (lind+k < elem_per_group) ) { v = d_inp[gind]; } else { v = NE; }
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
            if( (gind < N) && (lind+k < elem_per_group) ) {
                d_out[gind] = locmem[lind];
            }
        }
    }
}

