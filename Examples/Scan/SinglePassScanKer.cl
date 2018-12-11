typedef int     int32_t;
typedef uint    uint32_t;
typedef uchar   uint8_t;

#define WARP (1<<lgWARP)
#include "GenericHack.h"

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


enum { 
    STATUS_X = 0,
    STATUS_A = 1,
    STATUS_P = 2
}; 

inline uint8_t getStatus(uint8_t usd_flg) { return usd_flg &  3; }
inline uint8_t getUsed  (uint8_t usd_flg) { return usd_flg >> 2; }
inline uint8_t mkStatusUsed(uint8_t usd, uint8_t flg) { return (usd << 2) + flg; }

// Warp-reduce over the values, using an operator that does
// the following:
// op((flag1, value1, used1), (flag2, value2, used2)) =
//      if (flag2 == X)
//          return (X, value2, used2)
//      if (flag2 == P)
//          return (P, value2, dontcare);
//      flag2 is A
//      if (flag1 == X)
//          return (X, operator(value1, value2), used1+used2)
//      if (flag1 == P)
//          return (P, operator(value1, value2), dontcare)
//      flag1, flag2 is A
//      return (A, operator(value1, value2), used1+used2)
//
// which can be reduced to
//
// if (flag2 != A)  
//     return (flag2, value2, used2)
// return (flag1, operator(value1, value2), used1+used2)
inline void
binOpInLocMem( __local volatile ElTp*    sh_data
             , __local volatile uint8_t* sh_status
             , const size_t acc_th,  const size_t cur_th
) {
    ElTp agg1 = sh_data[acc_th];
    ElTp agg2 = sh_data[cur_th];
    uint8_t usd1, usd2, stat1, stat2;
    uint8_t tmp = sh_status[acc_th];
    stat1 = getStatus(tmp); 
    usd1  = getUsed(tmp);
    tmp   = sh_status[cur_th];    
    stat2 = getStatus(tmp);

    if (stat2 != STATUS_A) {
        agg1 = NE;
        usd1 = 0;
        stat1 = stat2;
    }
    usd1 += getUsed(tmp);
    sh_status[cur_th] = mkStatusUsed(usd1, stat1);
    sh_data[cur_th] = binOp(agg1, agg2);
}


inline void
flgBinOpInLocMem( __local volatile ElTp*    sh_data
                , __local volatile uint8_t* sh_status
                , __local volatile uint8_t* sh_flags
                , const size_t acc_th,  const size_t cur_th
) {
    uint8_t flg1 = sh_flags[acc_th];
    uint8_t flg2 = sh_flags[cur_th];
    ElTp agg1 = sh_data[acc_th];
    ElTp agg2 = sh_data[cur_th];
    uint8_t usd1, usd2, stat1, stat2;
    uint8_t tmp = sh_status[acc_th];
    stat1 = getStatus(tmp); usd1 = getUsed(tmp);
    tmp = sh_status[cur_th];    
    stat2 = getStatus(tmp); usd2 = getUsed(tmp);

    if (stat2 != STATUS_A) {
        flg1 = 0;
        agg1 = NE;
        usd1 = 0;
        stat1 = stat2;
    }
    usd1 += usd2;
    sh_status[cur_th] = mkStatusUsed(usd1, stat1);
    {
        FlgTuple tup1; tup1.flg = flg1; tup1.val = agg1;
        FlgTuple tup2; tup2.flg = flg2; tup2.val = agg2;
        FlgTuple res = binOpFlg( tup1, tup2 );
        sh_flags[cur_th] = res.flg;
        sh_data[cur_th] = res.val;
    }
}


inline void
incSpecialScanWarp (  __local volatile ElTp* sh_data
                    , __local volatile uint8_t* sh_status
                    , const int32_t  lane
) {
    if( lane >= 1 ) binOpInLocMem( sh_data, sh_status, lane-1, lane );
    if( lane >= 2 ) binOpInLocMem( sh_data, sh_status, lane-2, lane );
    if( lane >= 4 ) binOpInLocMem( sh_data, sh_status, lane-4, lane );
    if( lane >= 8 ) binOpInLocMem( sh_data, sh_status, lane-8, lane );
#if WARP == 32
    if( lane >= 16) binOpInLocMem( sh_data, sh_status, lane-16, lane);
#endif
}

inline void
incSpecialSgmScanWarp ( __local volatile ElTp* sh_data
                      , __local volatile uint8_t* sh_status
                      , __local volatile uint8_t* sh_flags
                      , const size_t lane
) {
    if( lane >= 1 ) flgBinOpInLocMem( sh_data, sh_status, sh_flags, lane-1, lane );
    if( lane >= 2 ) flgBinOpInLocMem( sh_data, sh_status, sh_flags, lane-2, lane );
    if( lane >= 4 ) flgBinOpInLocMem( sh_data, sh_status, sh_flags, lane-4, lane );
    if( lane >= 8 ) flgBinOpInLocMem( sh_data, sh_status, sh_flags, lane-8, lane );
#if WARP == 32
    if( lane >= 16) flgBinOpInLocMem( sh_data, sh_status, sh_flags, lane-16, lane);
#endif
}


inline ElTp
incScanWarp(__local volatile ElTp* sh_data, const size_t th_id) { 
    const size_t lane = th_id & (WARP-1);
    if( lane >= 1 ) sh_data[th_id] = binOp( sh_data[th_id-1 ], sh_data[th_id] );
    if( lane >= 2 ) sh_data[th_id] = binOp( sh_data[th_id-2 ], sh_data[th_id] );
    if( lane >= 4 ) sh_data[th_id] = binOp( sh_data[th_id-4 ], sh_data[th_id] );
    if( lane >= 8 ) sh_data[th_id] = binOp( sh_data[th_id-8 ], sh_data[th_id] );
#if WARP == 32
    if( lane >= 16) sh_data[th_id] = binOp( sh_data[th_id-16], sh_data[th_id] );
#endif
    return sh_data[th_id];
}

inline FlgTuple
incSgmScanWarp  ( __local volatile uint8_t* sh_flag
                , __local volatile ElTp*    sh_data
                , const size_t th_id
) {
    const size_t lane = th_id & (WARP-1);
    if( lane >= 1 ) {
        //sh_data[th_id] = binOp( sh_data[th_id-1 ], sh_data[th_id] );
        FlgTuple tup1; tup1.flg = sh_flag[th_id-1 ]; tup1.val = sh_data[th_id-1 ];
        FlgTuple tup2; tup2.flg = sh_flag[th_id   ]; tup2.val = sh_data[th_id   ];
        FlgTuple tup3 = binOpFlg( tup1, tup2 );
        sh_flag[th_id] = tup3.flg; sh_data[th_id] = tup3.val;
    }
    if( lane >= 2 ) {
        //sh_data[th_id] = binOp( sh_data[th_id-2 ], sh_data[th_id] );
        FlgTuple tup1; tup1.flg = sh_flag[th_id-2 ]; tup1.val = sh_data[th_id-2 ];
        FlgTuple tup2; tup2.flg = sh_flag[th_id   ]; tup2.val = sh_data[th_id   ];
        FlgTuple tup3 = binOpFlg( tup1, tup2 );
        sh_flag[th_id] = tup3.flg; sh_data[th_id] = tup3.val;
    }
    if( lane >= 4 ) {
        //sh_data[th_id] = binOp( sh_data[th_id-4 ], sh_data[th_id] );
        FlgTuple tup1; tup1.flg = sh_flag[th_id-4 ]; tup1.val = sh_data[th_id-4 ];
        FlgTuple tup2; tup2.flg = sh_flag[th_id   ]; tup2.val = sh_data[th_id   ];
        FlgTuple tup3 = binOpFlg( tup1, tup2 );
        sh_flag[th_id] = tup3.flg; sh_data[th_id] = tup3.val;
    }
    if( lane >= 8 ) {
        //sh_data[th_id] = binOp( sh_data[th_id-8 ], sh_data[th_id] );
        FlgTuple tup1; tup1.flg = sh_flag[th_id-8 ]; tup1.val = sh_data[th_id-8 ];
        FlgTuple tup2; tup2.flg = sh_flag[th_id   ]; tup2.val = sh_data[th_id   ];
        FlgTuple tup3 = binOpFlg( tup1, tup2 );
        sh_flag[th_id] = tup3.flg; sh_data[th_id] = tup3.val;
    }
#if WARP == 32
    if( lane >= 16) {
        //sh_data[th_id] = binOp( sh_data[th_id-16], sh_data[th_id] );
        FlgTuple tup1; tup1.flg = sh_flag[th_id-16]; tup1.val = sh_data[th_id-16];
        FlgTuple tup2; tup2.flg = sh_flag[th_id   ]; tup2.val = sh_data[th_id   ];
        FlgTuple tup3 = binOpFlg( tup1, tup2 );
        sh_flag[th_id] = tup3.flg; sh_data[th_id] = tup3.val;
    }
#endif
    //return sh_data[th_id];
    FlgTuple res; res.flg = sh_flag[th_id]; res.val = sh_data[th_id]; return res;
}


inline void 
incScanGroup( __local volatile ElTp*   sh_data, const size_t tid) {
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
    // result is returned directly; share memory does not contain it!
}

inline FlgTuple 
incSgmScanGroup ( __local volatile uint8_t* sh_flag
                , __local volatile ElTp*    sh_data
                , const size_t tid
) {
    const size_t lane   = tid & (WARP-1);
    const size_t warpid = tid >> lgWARP;

    // perform scan at warp level
    FlgTuple res = incSgmScanWarp(sh_flag, sh_data, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    // if last thread in a warp, record it at the beginning of sh_data
    if ( lane == (WARP-1) ) { sh_flag[warpid] = res.flg; sh_data[warpid] = res.val; }
    barrier(CLK_LOCAL_MEM_FENCE);

    // first warp scans the per warp results (again)
    if( warpid == 0 ) incSgmScanWarp(sh_flag, sh_data, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    // accumulate results from previous step;
    if (warpid > 0) {
        FlgTuple prev;
        prev.flg = sh_flag[warpid-1];
        prev.val = sh_data[warpid-1];
        res = binOpFlg( prev, res );
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sh_flag[tid] = res.flg;
    sh_data[tid] = res.val;
    barrier(CLK_LOCAL_MEM_FENCE);
    // result is returned directly; share memory does not contain it!
    return res;
}  
 
__kernel void singlePassScanKer ( 
        uint32_t            N,
        __global ElTp*      d_inp,      // read-only,  [N]
        __global ElTp*      d_out,      // write-only, [N]
        volatile __global int32_t*   global_id,  // read-write, [1]
        volatile __global ElTp*      aggregates, // read-write, [num_groups]
        volatile __global ElTp*      incprefix,  // read-write, [num_groups]
        volatile __global uint8_t*   statusflgs, // read-write, [num_groups]
        __local  int32_t*   block_id,
        __local  ElTp*      exchange
//      , __local uint8_t*    warpscan
) { 
    const int32_t tid = get_local_id(0);

    if (tid == 0) {
        int32_t id  = atomic_add(&global_id[0], 1);
        statusflgs[id] = STATUS_X;
        block_id[0] = id;

        // last block resets the global_id index!
        int32_t elem_per_group = get_local_size(0)*ELEMS_PER_THREAD;
        int32_t last_block = (N + elem_per_group - 1) / elem_per_group - 1;
        if(id == last_block) {
            global_id[0] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int32_t id   = block_id[0];

    ElTp  chunk[ELEMS_PER_THREAD];
    { // Coalesced read from global-input 'data' into register 'chunk' by means of shared memory
        const int32_t block_offset = id * get_local_size(0) * ELEMS_PER_THREAD;
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            uint32_t gind = block_offset + i*get_local_size(0) + tid;
            ElTp v = (gind < N) ? d_inp[gind] : NE;
            exchange[i*get_local_size(0) + tid] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint32_t loc_offset = tid * ELEMS_PER_THREAD;
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            chunk[i] = exchange[loc_offset+i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Per-Thread scan
    ElTp acc = chunk[0];
    {
        #pragma unroll
        for (uint32_t i = 1; i < ELEMS_PER_THREAD; i++) {
            acc = binOp(acc, chunk[i]);
            chunk[i] = acc;
        }
        exchange[tid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Per-Group Scan
    incScanGroup(exchange, tid);
    int32_t ind;
    if(tid == 0) { ind = get_local_size(0)-1; } else { ind = tid-1; }
    acc = exchange[ind];
    barrier(CLK_LOCAL_MEM_FENCE);

    ElTp prefix = NE;
    // Compute prefix from previous blocks (ASSUMES GROUP SIZE MULTIPLE OF 32!)
    {
        if ( (id == 0) && (tid == 0) ) { // id 0, first warp, first lane
            incprefix[id] = acc;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            statusflgs[id] = STATUS_P;
            acc = NE;
        } else if ( (id != 0) && (tid < WARP) ) { // id != 0, first warp, all lanes
            const int32_t lane = tid & (WARP-1);
            // parallel lookback in last warp

            if ( lane == 0 ) { // first lane
                aggregates[id] = acc;
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                statusflgs[id] = STATUS_A;
            }
          uint8_t stat1 = statusflgs[id-1];
	  if (stat1 == STATUS_P) {
                if(lane==0) prefix = incprefix[id-1];
          //} else if ((stat1 == STATUS_A) && (id > 1) && (statusflgs[id-2] == STATUS_P) ) {
          //      if(lane==0) prefix = binOp(incprefix[id-2], incprefix[id-1]);
          } else {   
            int32_t read_offset = id - 32;
            int32_t LOOP_STOP = -100;
            __local uint8_t* warpscan = (__local uint8_t*)(exchange+WARP);
	    
            while (read_offset != LOOP_STOP) {
                int32_t read_i = read_offset + lane;

                // Read 32 flag/aggregate values in the warp
                ElTp    aggr = NE;
                uint8_t flag = STATUS_X;
                uint8_t used = 0;
                if (read_i >= 0) {
                    flag = statusflgs[read_i];
                    if (flag == STATUS_P) {
                        aggr = incprefix[read_i];
                    } else if (flag == STATUS_A) {
                        aggr = aggregates[read_i];
                        used = 1;
                    }
                }
                // init local data for warp-reduce
                exchange[lane]       = aggr;
                warpscan[lane]       = mkStatusUsed(used, flag);
                incSpecialScanWarp(exchange, warpscan, lane);

                if ( lane == 0 ) {
                    // read result from local data after warp reduce
                    uint8_t usedflg_val = warpscan[WARP-1];
                    used = getUsed  (usedflg_val);
                    flag = getStatus(usedflg_val);
                    aggr = exchange[WARP-1];

                    prefix = binOp(aggr, prefix);
                    read_offset -= used;
                    if (flag == STATUS_P)
                        read_offset = LOOP_STOP;
                    block_id[0] = read_offset;
                }
                mem_fence(CLK_LOCAL_MEM_FENCE);   // WITHOUT THIS FENCE IT DEADLOCKS!!!
                read_offset = block_id[0];
            }
          }
            if (lane == 0) { 
                // publish prefix an status for next workgroups
                incprefix[id] = binOp(prefix, acc);
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                statusflgs[id] = STATUS_P;
                // publish current prefix for the threads in the current workgroup
                exchange[0] = prefix;
                acc = NE;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id != 0) prefix = exchange[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    { // finally read and add prefix to every element in this workgroup
      // Coalesced write to global-output 'data' from register 'chunk' by means of shared memory
        ElTp myacc = binOp(prefix, acc);
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            exchange[tid*ELEMS_PER_THREAD+i] = binOp(myacc, chunk[i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int32_t block_offset = id * get_local_size(0) * ELEMS_PER_THREAD;
        #pragma unroll
        for (int32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            int32_t gind = block_offset + i*get_local_size(0) + tid;
            if (gind < N) {
                d_out[gind] = exchange[gind - block_offset];
            }
        }
    }
}



__kernel void singlePassSgmScanKer ( 
        uint32_t            N,
        uint32_t            num_blocks_pad,
        __global uint8_t*   d_flg,      // read-only,  [N]
        __global ElTp*      d_inp,      // read-only,  [N]
        __global ElTp*      d_out,      // write-only, [N]
        volatile __global int32_t*   global_id,  // read-write, [1]
        volatile __global ElTp*      aggregates, // read-write, [num_groups]
        volatile __global ElTp*      incprefix,  // read-write, [num_groups]
        volatile __global uint8_t*   statusflgs, // read-write, [num_groups]
        __local  int32_t*   block_id,
        __local  uint8_t*   exchflgs,
        __local  ElTp*      exchange
//      , __local  uint8_t*   warpscan
) { 
    const size_t tid = get_local_id(0);
    const int32_t lane = tid & (WARP-1);

    volatile __global uint8_t* aggr_flgs = statusflgs + num_blocks_pad;
    volatile __global uint8_t* incp_flgs = aggr_flgs  + num_blocks_pad;

    if (tid == 0) {
        int32_t id  = atomic_add(&global_id[0], 1);
        statusflgs[id] = STATUS_X;
        block_id[0] = id;

        // last block resets the global_id index!
        int32_t elem_per_group = get_local_size(0)*SGM_ELEMS_PER_THREAD;
        int32_t last_block = (N + elem_per_group - 1) / elem_per_group - 1;
        if(id == last_block) {
            global_id[0] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    const int32_t id   = block_id[0];

    FlgTuple chunk[SGM_ELEMS_PER_THREAD];
    { // Coalesced read from global-input 'data' into register 'chunk' by means of shared memory
        const int32_t block_offset = id * get_local_size(0) * SGM_ELEMS_PER_THREAD;
        #pragma unroll
        for (uint32_t i = 0; i < SGM_ELEMS_PER_THREAD; i++) {
            uint32_t gind = block_offset + i*get_local_size(0) + tid;
            ElTp v = (gind < N) ? d_inp[gind] : NE;
            exchange[i*get_local_size(0) + tid] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint32_t loc_offset = tid * SGM_ELEMS_PER_THREAD;
        #pragma unroll
        for (uint32_t i = 0; i < SGM_ELEMS_PER_THREAD; i++) {
            chunk[i].val = exchange[loc_offset+i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    { // Coalesced read from global-input 'flags' into register 'chunk_flgs' by means of shared memory
        const int32_t block_offset = id * get_local_size(0) * SGM_ELEMS_PER_THREAD;
        #pragma unroll
        for (uint32_t i = 0; i < SGM_ELEMS_PER_THREAD; i++) {
            uint32_t gind = block_offset + i*get_local_size(0) + tid;
            uint8_t v = (gind < N) ? d_flg[gind] : 0;
            exchflgs[i*get_local_size(0) + tid] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint32_t loc_offset = tid * SGM_ELEMS_PER_THREAD;
        #pragma unroll
        for (uint32_t i = 0; i < SGM_ELEMS_PER_THREAD; i++) {
            chunk[i].flg = exchflgs[loc_offset+i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Per-Thread scan
    FlgTuple acc = chunk[0];
    #pragma unroll
    for (uint32_t i = 1; i < SGM_ELEMS_PER_THREAD; i++) {            
        acc = binOpFlg(acc, chunk[i]);
        chunk[i] = acc;
    }
    exchflgs[tid] = acc.flg;
    exchange[tid] = acc.val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Per-Group Scan
    acc = incSgmScanGroup(exchflgs, exchange, tid);
    FlgTuple prev_acc;
    if(tid == 0) {
        prev_acc.flg = 0;
        prev_acc.val = NE; 
    } else { 
        prev_acc.flg = exchflgs[tid-1];
        prev_acc.val = exchange[tid-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    FlgTuple prefix;
    { prefix.flg = 0; prefix.val = NE; }

    // Compute prefix from previous blocks (ASSUMES GROUP SIZE MULTIPLE OF 32!)
    {
        if ( (id == 0) && (tid == get_local_size(0)-1) ) { // id 0, last warp, last lane
            incprefix[id] = acc.val;
            incp_flgs[id] = acc.flg;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            statusflgs[id] = STATUS_P;
        } else if ( (id != 0) && ( WARP >= (get_local_size(0)-tid) ) ) { // id != 0, last warp, all lanes
            // parallel lookback in last warp
            if ( lane == (WARP-1) ) { // last lane
                aggregates[id] = acc.val;
                aggr_flgs[id]  = acc.flg;
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                statusflgs[id] = STATUS_A;
            }
            
            int32_t read_offset = id - 32;
            int32_t LOOP_STOP = -100;
            __local uint8_t* warpscan = (__local uint8_t*)(exchange+WARP);
            while (read_offset != LOOP_STOP) {
                int32_t read_i = read_offset + lane;

                // Read 32 flag/aggregate values in the warp
                ElTp    aggr = NE;
                uint8_t flag = 0;
                uint8_t stat = STATUS_X;
                uint8_t used = 0;
                if (read_i >= 0) {
                    stat = statusflgs[read_i];
                    if (stat == STATUS_P) {
                        aggr = incprefix[read_i];
                        flag = incp_flgs[read_i];
                    } else if (stat == STATUS_A) {
                        aggr = aggregates[read_i];
                        flag = aggr_flgs[read_i];
                        used = 1;
                    }
                }
                // init local data for warp-reduce
                exchflgs[lane]       = flag;
                exchange[lane]       = aggr;
                warpscan[lane]       = mkStatusUsed(used, stat);
                incSpecialSgmScanWarp(exchange, warpscan, exchflgs, lane);

                if ( lane == (WARP-1) ) {
                    { // update prefix
                        FlgTuple aggr; aggr.flg = exchflgs[lane]; aggr.val = exchange[lane];
                        prefix = binOpFlg(aggr, prefix);
                    }

                    { // update read_offset and publish it
                        uint8_t usedflg_val = warpscan[lane];
                        used = getUsed  (usedflg_val);
                        stat = getStatus(usedflg_val);
                        read_offset -= used;
                        if (stat == STATUS_P)
                            read_offset = LOOP_STOP;
                        block_id[0] = read_offset;
                    }
                }
                mem_fence(CLK_LOCAL_MEM_FENCE);   // WITHOUT THIS FENCE IT DEADLOCKS!!!
                read_offset = block_id[0];
            }

            if (lane == (WARP-1)) { 
                // publish prefix and status for next workgroups
                acc = binOpFlg(prefix, acc);
                incprefix[id] = acc.val;
                incp_flgs[id] = acc.flg;
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                statusflgs[id] = STATUS_P;
                // publish current prefix for the threads in the current workgroup
                exchflgs[0] = prefix.flg;
                exchange[0] = prefix.val;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id != 0) {
            prefix.flg = exchflgs[0];
            prefix.val = exchange[0];
        }
    }

    { // finally read and add prefix to every element in this workgroup
      // Coalesced write to global-output 'data' from register 'chunk' by means of shared memory
        acc = binOpFlg(prefix, prev_acc);        
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (uint32_t i = 0; i < SGM_ELEMS_PER_THREAD; i++) {
            FlgTuple res = binOpFlg(acc, chunk[i]);
            exchange[tid*SGM_ELEMS_PER_THREAD+i] = res.val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int32_t block_offset = id * get_local_size(0) * SGM_ELEMS_PER_THREAD;
        #pragma unroll
        for (int32_t i = 0; i < SGM_ELEMS_PER_THREAD; i++) {
            int32_t gind = block_offset + i*get_local_size(0) + tid;
            if (gind < N) {
                d_out[gind] = exchange[gind - block_offset];
            }
        }
    }
}
