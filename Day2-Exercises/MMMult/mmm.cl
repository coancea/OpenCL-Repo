#ifndef MMM_KERNELS
#define MMM_KERNELS

typedef int     int32_t;
typedef uint    uint32_t;
typedef uchar   uint8_t;

__kernel void naiveMMM  ( __global real* A
                        , __global real* B
                        , __global real* C
                        , uint32_t heightA
                        , uint32_t  widthB
                        , uint32_t  widthA
) {
    real accum = 0.0;
    uint32_t gidx = get_global_id(0);
    uint32_t gidy = get_global_id(1);
    if ( (gidx >= widthB) || (gidy >= heightA) ) return;

    for(uint32_t k = 0; k < widthA; k++) {
        accum += A[gidy*widthA + k] * B[k*widthB + gidx];
    }

    C[gidy*widthB + gidx] = accum;
}

/**
 * Implement block-tile matrix-matrix multiplication kernel below.
 * TILE is the tile size (e.g., 16)
 * For local-memory barriers use: "barrier(CLK_LOCAL_MEM_FENCE);"
 */ 
__kernel void blockMMM  ( __global real* A
                        , __global real* B
                        , __global real* C
                        , uint32_t heightA
                        , uint32_t  widthB
                        , uint32_t  widthA
) {
    __local real Ash[TILE][TILE];
    __local real Bsh[TILE][TILE]; 
    // add implementation here
}

/**
 * Implement register+block tiling for the matrix-matrix multiplication kernel below.
 * RT is the tile size (e.g., 16)
 * For local-memory barriers use: "barrier(CLK_LOCAL_MEM_FENCE);"
 */ 
__kernel void rgblkMMM  ( __global real* A
                        , __global real* B
                        , __global real* C
                        , uint32_t heightA
                        , uint32_t  widthB
                        , uint32_t  widthA
) { 
    __local real Ash[RT][RT+1];
    real cs[RT]; 
    // add implementation here
}
#endif //MMM_KERNELS
