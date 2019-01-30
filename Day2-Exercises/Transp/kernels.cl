#ifndef TRANSP_KERNELS
#define TRANSP_KERNELS

typedef int     int32_t;
typedef uint    uint32_t;
typedef uchar   uint8_t;

real arithmFun(real accum, real a) {
    return (a*a - accum);  // sqrt(accum) + a*a;
}

__kernel void memcpy_simple ( 
        uint32_t            N,
        __global real*      d_inp,      // read-only,  [N]
        __global real*      d_out       // write-only, [N]
) {
    uint32_t gid = get_global_id(0);
    if(gid < N) {
        d_out[gid] = d_inp[gid];
    }
}

__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
__kernel void naiveTransp( __global real* A
                         , __global real* B
                         , uint32_t height
                         , uint32_t width 
) {                                   // Can also be written as:
    uint32_t gidx = get_global_id(0); // get_group_id(0)*TILE + get_local_id(0);
    uint32_t gidy = get_global_id(1); // get_group_id(1)*TILE + get_local_id(1);

    if( (gidx >= width) || (gidy >= height) ) return;

    B[gidx*height + gidy] = A[gidy*width + gidx];
}

/**
 * Exercise 1 (Coalesced transposition): 
 *    ToDo: Implement the kernel below such that both
 *          reads and writes are coalesced.
 *    Note: "height" and "width" are the height and 
 *          width of "A" not of "B" (transpose of "A")!
 */
__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
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
    if( (gidx >= width) || (gidy >= height) ) return;
    B[gidx*height+gidy] = 0;
} 

__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
__kernel void optimTransp( __global real* A
                         , __global real* B
                         , uint32_t height 
                         , uint32_t width  
                         , __local  real* lmem // size of lmem: CHUNK * TILE * (TILE+1)
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

__kernel __attribute__((reqd_work_group_size(PRG_WGSIZE, 1, 1)))
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
        accum = arithmFun(accum, tmpA);
        B[offset + j] = accum;
    }
}

/**
 * Exercise 2 (Optimize coalescing by transposition): 
 *    ToDo: Implement the kernel below that reads/writes
 *          from/to the transposed version of A and B!
 *    Note: "height" and "width" are the height and 
 *          width of "A" not of "Atr" (transpose of "A")!
 */
__kernel __attribute__((reqd_work_group_size(PRG_WGSIZE, 1, 1)))
__kernel void coalsProgrm( __global real* Atr
                         , __global real* Btr
                         , uint32_t height
                         , uint32_t width 
) {
}

/**
 * Exercise 3 (Optimize coalescing by transposition): 
 *    ToDo: complete the implementation of the kernel below which
 *          processes a row of "A" per thread by streaming it,
 *          i.e., each threads processes CHUNK elements on its row
 *          of "A" at a time by:
 *          1. collectively reading "A" from global to local memory
 *             in coalesced fashion (IMPLEMENT),
 *          2. performing the original computation on its chunk, whose
 *             result is written to local memory ("CHUNK" elements of "B");
 *             THIS IS ALREADY IMPLEMENTED -- nothing to do here!
 *          3. collectively writing to global memory (of "B") in coalesced
 *             fashion (IMPLEMENT)
 *    Notes:
 *    "PRG_WGSIZE" denotes the local workgroup size, and is a multiple of "CHUNK".
 *    For best performance "CHUNK=16" since it results in fully coalesced access.
 *    However, 16 words of local memory per thread will likely affect occupancy
 *      (negatively).
 */
__kernel __attribute__((reqd_work_group_size(PRG_WGSIZE, 1, 1)))
__kernel void optimProgrm( __global real* A
                         , __global real* B
                         , uint32_t height
                         , uint32_t width
) {
    // scratchpad "fast" memory
	volatile __local real lmem[PRG_WGSIZE][CHUNK+1]; 
    uint32_t gid        = get_global_id(0);
    uint32_t lid        = get_local_id(0);
    uint32_t chunk_lane = lid % CHUNK;
    uint32_t tot_num_elem = width * height;

    real     accum  = 0.0;
    uint32_t offs_y = get_group_id(0)*get_local_size(0) + (lid/CHUNK);
    uint32_t step_y = get_local_size(0) / CHUNK;

    // In each iteration, each thread processes CHUNK elements of its row (of A/B)
    for(uint32_t j = 0; j < width; j+=CHUNK) {

        { // 1. read from "A" in coalesced fasion and place the result in local memory
          //    Implement this loop by writing proper computation instead of dummies!
            #pragma unroll
            for(uint32_t k = 0, ind_y = offs_y; k < CHUNK; k++, ind_y+=step_y) {
                uint32_t dummy = 0;
                uint32_t flat_loc_index = dummy;
                uint32_t ind_x = dummy;
                lmem[dummy][dummy] =
                    ((dummy < height) && (dummy < width)) ? A[dummy*width + dummy] : 0.0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        { // 2. execute original but with reads/writes from/to local memory
          //    NOTHING TO DO HERE: already implemented!
            #pragma unroll
            for(uint32_t k = 0; k < CHUNK; k++) {
                if( (j + k < width) && (gid < height)) {
                    real tmpA = lmem[lid][k]; //A[gid*width+j];
                    accum = arithmFun(accum, tmpA);
                    //B[gid*width + j + k] = accum;
                    lmem[lid][k] = accum;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        { // 3. read from "A" in coalesced fasion and place the result in local memory
          //    Implement this loop by writing proper computation instead of dummies!
          //    This is very similar to 1.
            #pragma unroll
            for(uint32_t k = 0, ind_y = offs_y; k < CHUNK; k++, ind_y+=step_y) {
                uint32_t dummy = 0;
                uint32_t flat_loc_index = dummy;
                uint32_t ind_x = dummy;
                real tmp = lmem[dummy][dummy];

                if ((dummy < height) && (dummy < width))
                    B[dummy*width + dummy] = tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#define CWAVE 16
__kernel void optimProgrm1( __global real* A
                         , __global real* B
                         , uint32_t height
                         , uint32_t width
) {
    // Assumes that GROUP-SIZE is PRG_WGSIZE and PRG_WGSIZE is a multiple of CWAVE
    volatile __local  real lmem[PRG_WGSIZE/2][CWAVE+1];
    real  dchunk[CWAVE];
    uint32_t gid        = get_global_id(0);
    uint32_t lid        = get_local_id(0);
    uint32_t chunk_lane = lid % CWAVE;
    uint32_t num_elem   = width * height;

    real     accum  = 0.0;
    uint32_t offs_y = (get_group_id(0)*get_local_size(0) + (lid/CWAVE)) * width;
    uint32_t step_y = (get_local_size(0)/CWAVE)*width;

    for(uint32_t j = 0; j < width; j+=CWAVE) {
        // load in shared memory
        uint32_t ind_x = chunk_lane + j;
        uint32_t ind_y = offs_y;
        #pragma unroll
        for(uint32_t r = 0; r < 2; r++) {
            #pragma unroll
            for(uint32_t k = 0; k < CWAVE/2; k++) {
                real tmp = 0.0;
                if ((ind_y < num_elem) && (ind_x < width))
                    tmp = A[ind_y + ind_x];
                //lmem[lid + k*get_local_size(0)] = tmp;
                uint32_t l_index = lid + k*get_local_size(0);
                lmem[l_index / CWAVE][l_index % CWAVE] = tmp;
                ind_y+=step_y;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            uint32_t half_group_size = get_local_size(0)/2;
            //uint32_t lidh = lid - r*half_group_size;
            for(uint32_t k = 0; k < CWAVE; k++) {
                if(lid >= r*half_group_size && lid < (r+1)*half_group_size)
                    dchunk[k] = lmem[lid-r*half_group_size][k];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // execute original but with reads/writes from register memory
        #pragma unroll
        for(uint32_t k = 0; k < CWAVE; k++) {
            if( (j + k < width) && (gid < height)) {
                real tmpA = dchunk[k]; //A[gid*width+j];
                accum = arithmFun(accum, tmpA);
                //B[gid*width + j + k] = accum;
                dchunk[k] = accum;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        ind_y = offs_y;
        // write back from register to local to global memory in coalesced fashion
        #pragma unroll
        for(uint32_t r = 0; r < 2; r++) {
            uint32_t half_group_size = get_local_size(0)/2;
            for(uint32_t k = 0; k < CWAVE; k++) {
                if(lid >= r*half_group_size && lid < (r+1)*half_group_size)
                    lmem[lid-r*half_group_size][k] = dchunk[k];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll
            for(uint32_t k = 0; k < CWAVE/2; k++) {
                if ((ind_y < num_elem) && (ind_x < width)) {
                    //real tmp = lmem[lid + k*get_local_size(0)];
                    uint32_t l_index = lid + k*get_local_size(0);
                    real tmp = lmem[l_index / CWAVE][l_index % CWAVE];
                    B[ind_y + ind_x] = tmp;
                }
                ind_y+=step_y;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}
#endif //TRANSP_KERNELS
