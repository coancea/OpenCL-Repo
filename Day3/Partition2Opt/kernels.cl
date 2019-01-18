typedef int     int32_t;
typedef uint    uint32_t;
typedef uchar   uint8_t;

#include "bridge.h" 

inline int2 binOp(const int2 a, const int2 b) {
    int2 res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    return res;
}

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

inline int2
incScanWarp(__local volatile int32_t* sh_data_x,
            __local volatile int32_t* sh_data_y,
            const size_t th_id
) { 
    const size_t lane = th_id & (WARP-1);
    #pragma unroll
    for(uint32_t i=0; i<lgWARP; i++) {
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
    const size_t lane   = tid & (WARP-1);
    const size_t warpid = tid >> lgWARP;

    // perform scan at warp level
    int2 res = incScanWarp(sh_data_x, sh_data_y, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    // if last thread in a warp, record it at the beginning of sh_data
    if ( lane == (WARP-1) ) {
        sh_data_x[warpid] = res.x;
        sh_data_y[warpid] = res.y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // first warp scans the per warp results (again)
    if( warpid == 0 ) incScanWarp(sh_data_x, sh_data_y, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    // accumulate results from previous step;
    if (warpid > 0) {
        int2 prev; 
        prev.x = sh_data_x[warpid-1];
        prev.y = sh_data_y[warpid-1];
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
        int2 tf = 0;
        uint32_t gid = group_offset + k + tid;
        if ( gid < N ) { 
            ElTp elm = d_inp[gid];
            tf.x = pred(elm);
            tf.y = 1 - tf.x;
        }
        acc = binOp(acc, tf);
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
    }
}

__kernel void scanPhaseKer ( 
        uint32_t            N,
        uint32_t            elem_per_group,
        __global ElTp*      d_inp,      // read-only,   [N]
        __constant int32_t* d_accT,     // read-only,   [number of workgroups]
        __constant int32_t* d_accF,     // read-only,   [number of workgroups]
        __global ElTp*      d_out,      // write-only,  [N]
        volatile __local int32_t* locmem  // local memory [group-size * ( 2 * sizeof(int32_t)+sizeof(ElTp) )];
) {
    volatile __local int32_t* locmem_x = locmem;
    volatile __local int32_t* locmem_y = locmem_x + get_local_size(0);
    volatile __local ElTp*    locmem_e = (volatile __local ElTp*) (locmem_y + get_local_size(0));

    const uint32_t tid = get_local_id(0);
    const uint32_t group_offset = get_group_id(0) * elem_per_group;
    int2 acc;
    uint32_t gii = d_accT[get_num_groups(0)-1];
    int2    accum = 0;
    if( get_group_id(0) > 0 ) {
        accum.x = d_accT[get_group_id(0)-1];
        accum.y = d_accF[get_group_id(0)-1];
    }

    for(uint32_t k = 0; k < elem_per_group; k += get_local_size(0)) {
        ElTp  elm;
        char  cond;
        int2  tf; tf.x = 0; tf.y = 0;
        uint32_t gid = group_offset + k + tid;
        if ( gid < N ) { 
            elm  = d_inp[gid];
            tf.x = pred(elm);
            tf.y = 1 - tf.x;
            cond = (char)tf.x;
        }
        
        // write to local memory and perform the intra-group scan
        locmem_x[tid] = tf.x;
        locmem_y[tid] = tf.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        tf = incScanGroup(locmem_x, locmem_y, tid);

        //compute the index in the local and global memory for the current element
        int32_t lmem_index, gmem_index;
        int2 last; 
        last.x = locmem_x[get_local_size(0)-1];
        last.y = locmem_y[get_local_size(0)-1];
        if( gid < N ) {
            if(cond == 1) { 
                lmem_index = tf.x - 1;
                gmem_index = accum.x + lmem_index;
            } else { 
                lmem_index = tf.y - 1;
                gmem_index = accum.y + tf.y - 1 + gii;
                lmem_index = lmem_index + last.x;
            }
        }
        // finally, write to global memory
#if 0
        if( gid < N ) {
            d_out[gmem_index] = elm;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#else
        // write in new order the element and its destination index to local memory
        if( gid < N ) {
            locmem_e[lmem_index] = elm;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ( gid < N ) {
            int32_t gmem_index;
            if (tid < last.x) { 
                gmem_index = (tid+accum.x); 
            } else {
                gmem_index = tid - last.x + accum.y + gii;
            }
            d_out[gmem_index] = locmem_e[tid];
        }
#endif
        // update accum for the next iteration
        accum.x += last.x; accum.y += last.y;
    }
}

