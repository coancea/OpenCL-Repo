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
            lmem[i*TILE*(TILE+1) + lidy*(TILE+1) + lidx] = A[gidy*width + gidx];
        gidy += TILE;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    gidx = get_group_id(1) * CHUNK * TILE + lidx;
    gidy = get_group_id(0) * TILE + lidy;

    #pragma unroll
    for (i = 0; i < CHUNK; i++) { 
        if( gidx < height && gidy < width )
            B[gidy*height + gidx] = lmem[i*TILE*(TILE+1) + lidx*(TILE+1) + lidy];
        gidx += TILE;
    }
}

__kernel void naiveProgrm( __global real* A
                         , __global real* B
                         , uint32_t height
                         , uint32_t width 
) {
    uint32_t gid = get_global_id(0); 

    if( gid >= height ) return;

    real     accum  = 0.0;
    uint32_t offset = gid * width;

    for(uint32_t j = 0; j < width; j++) {
        real tmpA = A[offset+j];
        accum = sqrt(accum) + tmpA*tmpA;
        B[offset + j] = accum;
    }
}

__kernel void coalsProgrm( __global real* A
                         , __global real* B
                         , uint32_t height
                         , uint32_t width 
) {
    uint32_t gid = get_global_id(0); 
    real     accum  = 0.0;

    if( gid >= height ) return;

    for(uint32_t j = 0; j < width; j++,gid+=height) {
        real tmpA = A[gid];
        accum = sqrt(accum) + tmpA*tmpA;
        B[gid] = accum;
    }
}

__kernel void optimProgrm( __global real* A
                         , __global real* B
                         , uint32_t height
                         , uint32_t width
                         , __local  real* lmem // size of lmem: workgroup_size * CHUNK
) {
    uint32_t gid        = get_global_id(0);
    uint32_t lid        = get_local_id(0);
    uint32_t chunk_lane = lid % CHUNK;
    uint32_t num_elem   = width * height;

    //if( gid >= height ) return;

    real     accum  = 0.0;
    uint32_t offs_y = (get_group_id(0)*get_local_size(0) + (lid/CHUNK)) * width;
    uint32_t step_y = (get_local_size(0)/CHUNK)*width;

    for(uint32_t j = 0; j < width; j+=CHUNK) {
        // load in shared memory
        #pragma unroll
        for(uint32_t k = 0, ind_y = offs_y; k < CHUNK; k++, ind_y+=step_y) {
            real tmp = 0.0;
            uint32_t ind_x = chunk_lane + j;
            if ((ind_y < num_elem) && (ind_x < width))
                tmp = A[ind_y + ind_x];
            lmem[lid + k*get_local_size(0)] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);  

        // execute original but with reads/writes from local memory
        #pragma unroll
        for(uint32_t k = 0; k < CHUNK; k++) {
            if( (j + k < width) && (gid < height)) {
                real tmpA = lmem[lid*CHUNK + k]; //A[gid*width+j];
                accum = sqrt(accum) + tmpA*tmpA;
                //B[gid*width + j + k] = accum;
                lmem[lid*CHUNK + k] = accum;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // write back from local to global memory in coalesced fashion
        #pragma unroll
        for(uint32_t k = 0, ind_y = offs_y; k < CHUNK; k++, ind_y+=step_y) {
            uint32_t ind_x = chunk_lane + j;
            real tmp = lmem[lid + k*get_local_size(0)];
            if ((ind_y < num_elem) && (ind_x < width))
                B[ind_y + ind_x] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#endif //TRANSP_KERNELS
