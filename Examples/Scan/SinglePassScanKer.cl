typedef int     int32_t;
typedef uint    uint32_t;
typedef uchar   uint8_t;
 
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

inline uint8_t  getLower16(const uint32_t w) { return (uint8_t)(w & ((1<<16)-1)); }
inline uint8_t  getUpper16(const uint32_t w) { return (uint8_t)(w >> 16); }
inline uint32_t mkFlStComp(const uint8_t flg, const uint8_t stat) {
    uint32_t res = flg; res = (res << 16); res += stat; return res;
}

inline uint8_t getStatus(const uint8_t usd_flg) { return usd_flg &  3; }
inline uint8_t getUsed  (const uint8_t usd_flg) { return usd_flg >> 2; }
inline uint8_t mkStatusUsed(const uint8_t usd, const uint8_t flg) { return (usd << 2) + flg; }

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
    uint8_t usd1, stat1, stat2;
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
    FlgTuple tup1; tup1.flg = sh_flags[acc_th]; tup1.val = sh_data[acc_th];
    FlgTuple tup2; tup2.flg = sh_flags[cur_th]; tup2.val = sh_data[cur_th];
    uint8_t usd1, stat1, stat2;
    uint8_t tmp = sh_status[acc_th];
    stat1 = getStatus(tmp); usd1 = getUsed(tmp);
    tmp = sh_status[cur_th];
    stat2 = getStatus(tmp);
    if (stat2 != STATUS_A) {
        tup1.flg = 0;
        tup1.val = NE;
        usd1     = 0;
        stat1    = stat2;
    }
    usd1 += getUsed(tmp);
    sh_status[cur_th] = mkStatusUsed(usd1, stat1);
    {
        FlgTuple res = binOpFlg( tup1, tup2 );
        sh_flags[cur_th] = res.flg;
        sh_data [cur_th] = res.val;
    }
}


inline void
incSpecialScanWarp (  __local volatile ElTp* sh_data
                    , __local volatile uint8_t* sh_status
                    , const int32_t  lane
) {
    #pragma unroll
    for(uint32_t i=0; i<lgWARP; i++) {
        const uint32_t p = (1<<i);
        if( lane >= p ) binOpInLocMem( sh_data, sh_status, lane-p, lane );
    }
}

inline void
incSpecialSgmScanWarp ( __local volatile ElTp* sh_data
                      , __local volatile uint32_t* sh_flags
                      , const size_t lane
) {
    #pragma unroll
    for (int32_t i=0; i<lgWARP; i++) {
        int32_t p = 1 << i;
        if (lane >= p) {
            //flgBinOpInLocMem( sh_data, sh_flags, lane-p, lane );
            size_t cur_th = lane;
            size_t acc_th = lane - p;
            uint8_t usd1, stat1, stat2, tmp;
            FlgTuple tup1, tup2;
            { // get accumulator values
                uint32_t acc_fl_usst = sh_flags[acc_th];
                tup1.flg = getUpper16(acc_fl_usst);
                tmp = getLower16(acc_fl_usst);
                stat1 = getStatus(tmp); usd1 = getUsed(tmp);   
                tup1.val = sh_data[acc_th];
            }
            { // get current elem values
                uint32_t acc_fl_usst = sh_flags[cur_th];
                tup2.flg = getUpper16(acc_fl_usst);
                tmp = getLower16(acc_fl_usst);
                stat2 = getStatus(tmp); 
                tup1.val = sh_data[cur_th];
            }
            if (stat2 != STATUS_A) {
                tup1.flg = 0;
                tup1.val = NE;
                usd1     = 0;
                stat1    = stat2;
            }  
            usd1 += getUsed(tmp);
            stat1 = mkStatusUsed(usd1, stat1);
            {
                FlgTuple res = binOpFlg( tup1, tup2 );
                sh_flags[cur_th] = mkFlStComp(res.flg, stat1); 
                sh_data [cur_th] = res.val;
            }   
        }
    }
}


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

inline FlgTuple
incSgmScanWarp  ( __local volatile uint8_t* sh_flag
                , __local volatile ElTp*    sh_data
                , const size_t th_id
) {
    const size_t lane = th_id & (WARP-1);
    #pragma unroll
    for(uint32_t i=0; i<lgWARP; i++) {
        const uint32_t p = (1<<i);
        if( lane >= p ) {
            FlgTuple tup1; tup1.flg = sh_flag[th_id-p]; tup1.val = sh_data[th_id-p];
            FlgTuple tup2; tup2.flg = sh_flag[th_id  ]; tup2.val = sh_data[th_id  ];
            FlgTuple tup3 = binOpFlg( tup1, tup2 );
            sh_flag[th_id] = tup3.flg; sh_data[th_id] = tup3.val;
        }
    }
    FlgTuple res; res.flg = sh_flag[th_id]; res.val = sh_data[th_id]; return res;
}

inline void 
incScanGroup0 ( __local volatile ElTp* sh_data, const size_t th_id) {
    ElTp res;
    #pragma unroll
    for(uint32_t i=0; i<logWORKGROUP_SIZE; i++) {
        const uint32_t p = (1<<i);
        if ( th_id >= p ) res = binOp( sh_data[th_id-p], sh_data[th_id] );
        barrier(CLK_LOCAL_MEM_FENCE);
        if ( th_id >= p ) sh_data[th_id] = res;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}  
 
inline void 
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
}

inline void 
incSgmScanGroup ( __local volatile uint8_t* sh_flag
                , __local volatile ElTp*    sh_data
                , const size_t th_id
) {
    FlgTuple acc, inp;
    #pragma unroll
    for(uint32_t i=0; i<logWORKGROUP_SIZE; i++) {
        const uint32_t p = (1<<i);
        if ( th_id >= p ) {
            acc.flg = sh_flag[th_id-p]; acc.val = sh_data[th_id-p];
            inp.flg = sh_flag[th_id  ]; inp.val = sh_data[th_id  ];
            inp = binOpFlg( acc, inp );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ( th_id >= p ) {
            sh_flag[th_id] = inp.flg;
            sh_data[th_id] = inp.val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void 
incSgmScanGroup0( __local volatile uint8_t* sh_flag
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
}  

 
inline void
warpScanSpecial ( __local volatile uint8_t* sh_flag
                , __local volatile ElTp*    sh_data
                , const size_t th_id
) {
    const size_t lane = th_id & (WARP-1);
    #pragma unroll
    for(uint32_t i=0; i<lgWARP; i++) {
        const uint32_t p = (1<<i);
        if( lane >= p ) {
            uint8_t f1 = sh_flag[th_id-p]; ElTp v1 = sh_data[th_id-p];
            uint8_t f2 = sh_flag[th_id  ]; ElTp v2 = sh_data[th_id  ];

            uint8_t f; ElTp v;
            if(f2 == 2 || f2 == 0) { f = f2; v = v2;}
            else                   { f = f1; v = v1 + v2; }

            sh_flag[th_id] = f; sh_data[th_id] = v;
        }
    }
}
__kernel void singlePassScanKer ( 
        uint32_t            N,
        __global ElTp*      d_inp,      // read-only,  [N]
        __global ElTp*      d_out,      // write-only, [N]
        volatile __global int32_t*   global_id,  // read-write, [1]
        volatile __global ElTp*      aggregates, // read-write, [num_groups]
        volatile __global ElTp*      incprefix,  // read-write, [num_groups]
        volatile __global uint8_t*   statusflgs, // read-write, [num_groups]
        volatile __local  int32_t* restrict  block_id,
        volatile __local  ElTp* restrict     exchange
) {
    const int32_t tid = get_local_id(0);

    if (tid == 0) {
        int32_t id  = atomic_add(&global_id[0], 1);
        statusflgs[id] = STATUS_X;
        block_id[0] = id;

        // last block resets the global_id index!
        int32_t elem_per_group = get_local_size(0) * ELEMS_PER_THREAD;
        int32_t last_block = (N + elem_per_group - 1) / elem_per_group - 1;
        if(id == last_block) {
            global_id[0] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int32_t WG_ID = block_id[0];

    ElTp  chunk[ELEMS_PER_THREAD];
    { // Coalesced read from global-input 'data' into register 'chunk' by means of shared memory
        const int32_t block_offset = WG_ID * get_local_size(0) * ELEMS_PER_THREAD;
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            uint32_t gind = block_offset + i*get_local_size(0) + tid;
            ElTp v = (gind < N) ? d_inp[gind] : NE;
            exchange[gind - block_offset] = v;
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

    { // Per-Group Scan, then store in local memory
        incScanGroup(exchange, tid);
        int32_t prev_ind = (tid == 0) ? (get_local_size(0) - 1) : (tid - 1);
        acc = exchange[prev_ind];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    ElTp prefix = NE;
    // Compute prefix from previous blocks (ASSUMES GROUP SIZE MULTIPLE OF 32!)
    {
        if ( (WG_ID == 0) && (tid == 0) ) { // first group, first warp, first lane
            incprefix[WG_ID] = acc;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            statusflgs[WG_ID] = STATUS_P;
            acc = NE;
        }
        if ( (WG_ID != 0) && (tid < WARP) ) { // WG_ID != 0, first warp, all lanes
            volatile __local uint8_t * warpscan = (volatile __local uint8_t*)(exchange+get_local_size(0));
            
            if ( tid == 0 ) { // first lane
                // publish the partial result a.s.a.p.
                aggregates[WG_ID] = acc;
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                statusflgs[WG_ID] = STATUS_A;
                warpscan[0] = statusflgs[WG_ID-1];
            }
            mem_fence(CLK_LOCAL_MEM_FENCE); 
            uint8_t stat1 = warpscan[0];
	        if (stat1 == STATUS_P) {
                // important performance optimization:
                // do not enter the expensive communication if
                // the previous workgroup has published its prefix!
                if(tid == 0) {
                    prefix = incprefix[WG_ID-1];
                }
            } else {
#if 0
                ElTp aggr = NE;
                mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
                if(tid == 0) {
                    bool goOn = true;
                    while(goOn) {
                        aggr = NE;
                        int index = WG_ID - 1;
                        //mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 
                        volatile uint8_t flg = statusflgs[index];
                        
                        while ((flg != STATUS_A) && (index > 0)) {
                            aggr = binOp(aggr, incprefix[index]);
                            index--;
                            //mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 
                            flg = statusflgs[index];
                        }
                        if (statusflgs[index] == STATUS_P) goOn = false;
                        
                        //mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 
                    }
                    prefix = aggr;
                    //mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
                }
                mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
#else
                int32_t read_offset = WG_ID - WARP;
                int32_t LOOP_STOP = -WARP;

	            // look at WARP previous blocks at a time until
                // the whole prefix of this workgroup is computed
                while (read_offset > LOOP_STOP) {
                    int32_t read_i = read_offset + tid;

                    // Read WARP flag/aggregate values in local memory
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
#if 1
                    exchange[tid]       = aggr;
                    warpscan[tid]       = flag;

                    if(warpscan[WARP-1] != STATUS_P)
                        warpScanSpecial ( warpscan, exchange, tid);
                    flag = warpscan[WARP-1];
                    aggr = exchange[WARP-1];

                    // now we have performed the scan; advance only if flag is not 0
                    if (flag == STATUS_P) {
                        read_offset = LOOP_STOP;
                    } else if (flag == STATUS_A) {
                        read_offset = read_offset - WARP;
                    }
                    if (flag != STATUS_X) {
                        prefix = binOp(aggr, prefix);             
                    }

                    mem_fence(CLK_LOCAL_MEM_FENCE);

#else
                    exchange[tid]       = aggr;
                    warpscan[tid]       = mkStatusUsed(used, flag);
                    mem_fence(CLK_LOCAL_MEM_FENCE);
                    // perform reduce
                    if(warpscan[WARP-1] != STATUS_P)
                        incSpecialScanWarp(exchange, warpscan, tid);
                    mem_fence(CLK_LOCAL_MEM_FENCE);
                    if ( tid == 0 ) {
                        // read result from local data after warp reduce
                        uint8_t usedflg_val = warpscan[WARP-1];
                        flag = getStatus(usedflg_val);
                        if (flag == STATUS_P) { // LOOP WILL BE EXITED
                            read_offset = LOOP_STOP;
                        } else { // LOOP WILL CARRY ON
                            used = getUsed  (usedflg_val);
                            read_offset = read_offset - used;
                        }
                        block_id[0] = read_offset;
                        // update prefix with the current contribution
                        aggr = exchange[WARP-1];
                        prefix = binOp(aggr, prefix);
                    }
                    mem_fence(CLK_LOCAL_MEM_FENCE);
                    read_offset = block_id[0];
#endif
                } // END WHILE loop
#endif
            } // END ELSE branch of if (stat1 == STATUS_P)

            if(tid == 0) {
                // publish the prefix of the current workgroup
                incprefix[WG_ID] = binOp(prefix, acc);
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                statusflgs[WG_ID] = STATUS_P;
                // publish prefix for all work items and update acc
                exchange[0] = prefix;
                acc = NE;
            }
        } // end IF(WG_ID != 0) && (tid < WARP)

        if (WG_ID != 0)  {
            // all workgroup threads read the prefix
            barrier(CLK_LOCAL_MEM_FENCE);
            prefix = exchange[0];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    { // finally read and add prefix to every element in this workgroup
      // Coalesced write to global-output 'data' from register 'chunk' by means of shared memory
        ElTp myacc = binOp(prefix, acc);
        #pragma unroll
        for (uint32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            exchange[tid*ELEMS_PER_THREAD+i] = binOp(myacc, chunk[i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const uint32_t block_offset = WG_ID * get_local_size(0) * ELEMS_PER_THREAD;
        #pragma unroll
        for (int32_t i = 0; i < ELEMS_PER_THREAD; i++) {
            uint32_t gind = block_offset + i * get_local_size(0) + tid;
            if (gind < N) {
                d_out[gind] = exchange[gind - block_offset];
            }
        }
    }
}
 
__kernel void singlePassSgmScanKer ( 
        uint32_t            N,
        __global uint8_t*   d_flg,      // read-only,  [N]
        __global ElTp*      d_inp,      // read-only,  [N]
        __global ElTp*      d_out,      // write-only, [N]
        volatile __global int32_t*   global_id,  // read-write, [1]
        volatile __global ElTp*      aggregates, // read-write, [num_groups]
        volatile __global ElTp*      incprefix,  // read-write, [num_groups]
        volatile __global uint8_t*   statusflgs, // read-write, [num_groups]
        __local  int32_t* block_id,
        __local  ElTp*    exchange
) { 
    const size_t  tid = get_local_id(0);

    if (tid == 0) {
        int32_t id = atomic_add(&global_id[0], 1);
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

    const int32_t WG_ID = block_id[0];

    FlgTuple chunk[SGM_ELEMS_PER_THREAD];
    { // Coalesced read from global-input 'data' into register 'chunk' by means of shared memory
        const uint32_t block_offset = WG_ID * get_local_size(0) * SGM_ELEMS_PER_THREAD;
        #pragma unroll
        for (uint32_t i = 0; i < SGM_ELEMS_PER_THREAD; i++) {
            uint32_t gind = block_offset + i*get_local_size(0) + tid;
            ElTp v = (gind < N) ? d_inp[gind] : NE;
            exchange[gind - block_offset] = v;
        }
        uint32_t loc_offset = tid * SGM_ELEMS_PER_THREAD;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (uint32_t i = 0; i < SGM_ELEMS_PER_THREAD; i++) {
            chunk[i].val = exchange[loc_offset+i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    { // Coalesced read from global-input 'flags' into register 'chunk_flgs'
      // by means of shared memory. For performance reasons, work with `uint32_t`
      // rather than `uint8_t` in local memory, in order to reduce conflicts.
        volatile __local uint32_t* restrict exchflgs = (__local  uint32_t*) exchange;
        const uint32_t block_offset = WG_ID * get_local_size(0) * SGM_ELEMS_PER_THREAD;
        #pragma unroll
        for (uint32_t i = 0; i < SGM_ELEMS_PER_THREAD; i++) {
            uint32_t gind = block_offset + i*get_local_size(0) + tid;
            uint8_t v = (gind < N) ? d_flg[gind] : 0;
            exchflgs[gind - block_offset] = v;
        }
        uint32_t loc_offset = tid * SGM_ELEMS_PER_THREAD;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (uint32_t i = 0; i < SGM_ELEMS_PER_THREAD; i++) {
            chunk[i].flg = (uint8_t)exchflgs[loc_offset+i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    FlgTuple acc = chunk[0];
    { // Per-Thread scan, then store in local memory
        #pragma unroll
        for (uint32_t i = 1; i < SGM_ELEMS_PER_THREAD; i++) {            
            acc = binOpFlg(acc, chunk[i]);
            chunk[i] = acc;
        }
    }

    // Per-Group Scan, then load the result
    {
        volatile __local  uint8_t* restrict exchflgs = (__local  uint8_t*) (exchange + get_local_size(0));
        exchflgs[tid] = acc.flg;
        exchange[tid] = acc.val;

        barrier(CLK_LOCAL_MEM_FENCE);
        incSgmScanGroup(exchflgs, exchange, tid);

        int32_t prev_ind = (tid == 0) ? (get_local_size(0)-1) : (tid-1);
        acc.flg = exchflgs[prev_ind];
        acc.val = exchange[prev_ind];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    FlgTuple prefix;
    { prefix.flg = 0; prefix.val = NE; }

    // Compute prefix from previous blocks (ASSUMES GROUP SIZE MULTIPLE OF WARP!)
    { 
        volatile __local  uint32_t* restrict exchflgs = (__local  uint32_t*) (exchange + get_local_size(0));
        if ( (WG_ID == 0) && (tid == 0) ) { // first workgroup, first thread
            incprefix[WG_ID] = acc.val;
            if (acc.flg > 0) { acc.flg = 1; }
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            statusflgs[WG_ID] = mkStatusUsed(acc.flg, STATUS_P);
            // reset accumulator
            acc.flg = 0; acc.val = NE;
        } 
        if ( (WG_ID != 0) && (tid < WARP) ) { // the other workgroups, first WARP
            if ( tid == 0 ) { // first thread
                // record this workgroup status a.s.a.p.
                aggregates[WG_ID] = acc.val;
                if (acc.flg > 0) { acc.flg = 1; } // COSMIN
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                statusflgs[WG_ID] = mkStatusUsed(acc.flg, STATUS_A);
                // read previous workgroup status for fasttrack opportunity
                exchflgs[0] = statusflgs[WG_ID-1];
            }
            mem_fence(CLK_LOCAL_MEM_FENCE); 
            uint8_t stat1 = (uint8_t)exchflgs[0];
	        if (getStatus(stat1) == STATUS_P) {
                // important performance optimization:
                // do not enter the expensive communication if
                // the previous workgroup has published its prefix!
                if(tid == 0) {
                    // publishing the prefix, reset accumulator
                    prefix.val = incprefix[WG_ID-1];
                    prefix.flg = getUsed(stat1);                    
                }
            } else {
                int32_t read_offset = WG_ID - WARP;
                int32_t LOOP_STOP = -100;
                while (read_offset != LOOP_STOP) {
                    int32_t read_i = read_offset + tid;

                    // Read WARP flag/aggregate values in the warp
                    ElTp    aggr = NE;
                    uint8_t flag = 0;
                    uint8_t stat = STATUS_X;
                    uint8_t used = 0;
                    if (read_i >= 0) {
                        uint8_t flg_stat = statusflgs[read_i];
                        stat = getStatus(flg_stat);
                        flag = getUsed(flg_stat);
                        if (stat == STATUS_P) {
                            aggr = incprefix[read_i];
                        } else if (stat == STATUS_A) {
                            aggr = aggregates[read_i];
                            used = 1;
                        } else { flag = 0; }
                    }
                    // init local data for warp-reduce
                    exchange[tid] = aggr;
                    exchflgs[tid] = mkFlStComp(flag, mkStatusUsed(used, stat));
                    mem_fence(CLK_LOCAL_MEM_FENCE);
                    if(getLower16(exchflgs[WARP-1]) != STATUS_P)
                        incSpecialSgmScanWarp(exchange, exchflgs, tid);
                    mem_fence(CLK_LOCAL_MEM_FENCE);
                    if ( tid == 0 ) {
                        // update read_offset
                        uint32_t tmp_flust = exchflgs[WARP-1];
                        uint8_t usedflg_val = getLower16(tmp_flust);
                        stat = getStatus(usedflg_val);
                        if (stat == STATUS_P) { // LOOP WILL EXIT
                            // publish exit condition
                            read_offset = LOOP_STOP;
                        } else {
                            used = getUsed(usedflg_val);
                            read_offset = read_offset - used;
                        }
                        block_id[0] = read_offset;

                        // update prefix 
                        FlgTuple aggr; 
                        aggr.flg = getUpper16(tmp_flust);
                        aggr.val = exchange[WARP-1];
                        prefix = binOpFlg(aggr, prefix);
                    }
                    mem_fence(CLK_LOCAL_MEM_FENCE);
                    read_offset = block_id[0];
                } // end WHILE loop
            } // end ELSE branch of IF(stat1 == STATUS_P)
            if(tid == 0) {
                // publish full prefix and status for next workgroups
                acc = binOpFlg(prefix, acc);
                incprefix[WG_ID] = acc.val;
                if (acc.flg > 0) { acc.flg = 1; }
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                statusflgs[WG_ID] = mkStatusUsed(acc.flg, STATUS_P);
                // reset acc
                acc.flg = 0; acc.val = NE;
                // publish current prefix for the threads in this workgroup
                exchflgs[0] = prefix.flg;
                exchange[0] = prefix.val;
            }
        } // END IF( (WG_ID != 0) && (tid < WARP) )
     
        if (WG_ID != 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (exchflgs[0] > 0) { prefix.flg = 1; } else { prefix.flg = 0; }
            prefix.val = exchange[0];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    } 
 
    { // finally read and add prefix to every element in this workgroup
      // Coalesced write to global-output 'data' from register 'chunk' by means of shared memory
        acc = binOpFlg(prefix, acc);
        const uint32_t local_offset = tid*SGM_ELEMS_PER_THREAD;
        #pragma unroll
        for (uint32_t i = 0; i < SGM_ELEMS_PER_THREAD; i++) {
            FlgTuple res = binOpFlg(acc, chunk[i]);
            exchange[local_offset+i] = res.val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint32_t block_offset = WG_ID * get_local_size(0) * SGM_ELEMS_PER_THREAD;
        #pragma unroll
        for (int32_t i = 0; i < SGM_ELEMS_PER_THREAD; i++) {
            uint32_t gind = block_offset + i*get_local_size(0) + tid;
            if (gind < N) {
                d_out[gind] = exchange[gind - block_offset];
            }
        }
    }
}
