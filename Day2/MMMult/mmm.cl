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

__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
__kernel void blockMMM  ( __global real* A
                        , __global real* B
                        , __global real* C
                        , uint32_t heightA
                        , uint32_t  widthB
                        , uint32_t  widthA
) {
    __local real Ash[TILE][TILE];
    __local real Bsh[TILE][TILE]; 

    real accum = 0.0;
    uint32_t gidx = get_global_id(0);
    uint32_t gidy = get_global_id(1);
    uint32_t lidx = get_local_id(0);
    uint32_t lidy = get_local_id(1);
    
    for(uint32_t kk = 0; kk < widthA; kk += TILE) {
        real tmp = 0.0;
        if ((gidy < heightA) && (kk+lidx < widthA))
            tmp = A[gidy*widthA + kk + lidx];
        Ash[lidy][lidx] = tmp;

        tmp = 0.0;
        if ((gidx < widthB)  && (kk+lidy < widthA)) 
            tmp = B[(lidy+kk)*widthB + gidx];
        Bsh[lidy][lidx] = tmp;
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for(int k = 0; k < TILE; k++)
            accum += Ash[lidy][k] * Bsh[k][lidx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if( (gidx < widthB) && (gidy < heightA) )
        C[gidy*widthB + gidx] = accum;
}

__kernel __attribute__((reqd_work_group_size(RT*RT, 1, 1)))
__kernel void rgblkMMM  ( __global real* A
                        , __global real* B
                        , __global real* C
                        , uint32_t heightA
                        , uint32_t  widthB
                        , uint32_t  widthA
) { 
    __local real Ash[RT][RT+1];
    real cs[RT]; 

    uint32_t ii   = get_group_id(1) * RT;
	uint32_t j    = get_global_id(0);
	uint32_t locx = get_local_id(0) % RT;
	uint32_t locy = get_local_id(0) / RT;

    #pragma unroll
    for(int i=0; i<RT; i++)
        cs[i] = 0.0;

    for(int kk = 0; kk < widthA; kk += RT) {
        real tmp = 0.0;
        if ((ii+locy < heightA) && (kk+locx < widthA)) {
            tmp = A[(ii+locy)*widthA + kk+locx];
        }
        Ash[locy][locx] = tmp;
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < RT; k++) {
            real b = 0.0;
            if ((k+kk < widthA) && (j < widthB)) {
                b = B[(k+kk)*widthB + j];
            }
            #pragma unroll 
            for(int i = 0; i < RT; i++) {
                cs[i] += Ash[i][k] * b;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for(int i=0; i<RT; i++) {
        if( (ii+i < heightA) && (j < widthB) )
            C[(ii+i)*widthB + j] = cs[i];
    }
}
#endif //MMM_KERNELS
