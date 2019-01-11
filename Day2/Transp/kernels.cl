#ifndef TRANSP_KERNELS
#define TRANSP_KERNELS

#include "bridge.h"

typedef int     int32_t;
typedef uint    uint32_t;
typedef uchar   uint8_t;

__kernel void naiveTransp( __global real* A
                         , __global real* B
                         , uint32_t height
                         , uint32_t width 
) {
    uint32_t gidx = get_global_id(0); 
    uint32_t gidy = get_global_id(1); 

    if( (gidx >= width) || (gidy >= height) ) return;

    B[gidx*height + gidy] = A[gidy*width + gidx];
}


__kernel void coalsTransp( __global real* A
                         , __global real* B
                         , uint32_t height
                         , uint32_t width  
                         , __local  real* lmem // size of lmem: TILE * (TILE+1)
) {
    uint32_t gidx = get_global_id(0);
    uint32_t lidx = get_local_id(0);
    uint32_t gidy = get_global_id(1);
    uint32_t lidy = get_local_id(1);

    if( gidx < width && gidy < height )
        lmem[lidy*(TILE+1) + lidx] = A[gidy*width + gidx];

    barrier(CLK_LOCAL_MEM_FENCE);

    gidx = get_group_id(1) * TILE + lidx; 
    gidy = get_group_id(0) * TILE + lidy;

    if( gidx < height && gidy < width )
        B[gidy*height + gidx] = lmem[lidx*(TILE+1) + lidy];
}

__kernel void optimTransp( __global real* A
                         , __global real* B
                         , uint32_t height 
                         , uint32_t width  
                         , __local  real* lmem // size of lmem: TILE * TILE * CHUNK
) {
    uint32_t lidx = get_local_id(0);
    uint32_t gidx = get_global_id(0); 
    uint32_t lidy = get_local_id(1);
    uint32_t gidy = get_group_id(1) * CHUNK * TILE + lidy;
    uint32_t i;
    #pragma unroll
    for (i = 0; i < CHUNK; i++) {
        if( gidx < width && gidy < height )
            lmem[i*TILE*(TILE+1) + lidy*(TILE+1) + lidx] = A[(gidy+i*TILE)*width + gidx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (i = 0; i < CHUNK; i++) { 
        gidx = get_group_id(1) * CHUNK * TILE + lidx;
        gidy = get_group_id(0) * TILE + lidy;

        if( gidx < height && gidy < width )
            B[gidy*height + (gidx+i*TILE)] = lmem[i*TILE*(TILE+1) + lidx*(TILE+1) + lidy];
    }
}

#endif //TRANSP_KERNELS
