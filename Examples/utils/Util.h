#ifndef GENERIC_UTILITIES
#define GENERIC_UTILITIES

/*******************************************************/
/*****  Utilities Related to Time Instrumentation  *****/
/*******************************************************/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#if (WITH_DOUBLE)
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
    typedef double         REAL;
    typedef unsigned long  ULONG;
#else
    typedef float        REAL;
    typedef unsigned int ULONG;
#endif

// Device-id, local/constant/global-memory size in KB, 
#define GPU_DEV_ID      0
#define GPU_LOC_MEM     48
#define GPU_CONST_MEM   64
#define GPU_GLB_MEM     2097152   

// Number of registers per core, preferred local-memory per thread
#define GPU_REG_MEM     32
#define GPU_LOC_MEM_THD 8
#define GPU_NUM_CORES   2880
        
// Utility functions
// CHR: helper function for computing time differences at microsecond resolution
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

bool is_pow2(int atr_val) {
    int x = 1;

    for(int i = 0; i < 31; i++) {
        if(x == atr_val) return true;
        x = (x << 1);
    }
    return false;
}

#endif //GENERIC_UTILITIES
