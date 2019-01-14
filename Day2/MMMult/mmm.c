#include "../../clutils.h"
#include <stdbool.h>
#include <math.h>
#include "bridge.h"
#include "mmm.h"

#define HEIGHT_A 1024 //537
#define WIDTH_A  256  //422
#define WIDTH_B  2048 //732

#define RUNS_CPU 1
#define RUNS_GPU 50

/*******************************/
/***        Helpers          ***/
/*******************************/

typedef enum { 
    NAIVE = 0,
    BLOCK = 1,
    RGBLK = 2
} Version;

void randomInit(real* data, const uint32_t size) {
    for (uint32_t i = 0; i < size; ++i)
        data[i] = rand() / (real)RAND_MAX;
}

void goldMMM(real* A, real* B, real* C, uint32_t rowsA, uint32_t colsB, uint32_t colsA) {
  int64_t bef = get_wall_time();

  for(int l = 0; l < RUNS_CPU; l++) {
    for(uint32_t i = 0; i < rowsA; i++) {
      for(uint32_t j = 0; j < colsB; j++) {
        float sum = 0.0;
        for(uint32_t k = 0; k < colsA; k++) {
          sum += A[i*colsA + k] * B[k * colsB + j];
        }
        C[i * colsB + j] = sum;
      }
    } 
  }

  int64_t aft = get_wall_time();
  int64_t elapsed_us = aft-bef;

  printf("Golden (sequential) dense matrix-matrix multiplication average runtime: %dμs (across %d runs)\n",
         (uint32_t)(elapsed_us/RUNS_CPU), RUNS_CPU);
}

bool validate(real* A, real* B, uint32_t sizeAB){
    for(uint32_t i = 0; i < sizeAB; i++)
      if (fabs(A[i] - B[i]) > 0.0005){
        printf("INVALID RESULT %d %f %f\n", i, A[i], B[i]);
        return false;
      }
    printf("VALID RESULT!\n");
    return true;
}

inline size_t mkGlobalDim(const uint32_t pardim, const uint32_t T) {
    return ((pardim + T - 1) / T) * T;
}

void runGPUverMMM(Version kind, real* hC, real* hdC) {
    size_t localWorkSize[2];
    size_t globalWorkSize[2];
    cl_kernel kernel;

    if ( (kind == NAIVE) || (kind == BLOCK) ) {
        kernel = (kind == NAIVE) ? kers.naiveMMM : kers.blockMMM;
        localWorkSize [0] = TILE;
        localWorkSize [1] = TILE;
        globalWorkSize[0] = mkGlobalDim(buffs.widthB , TILE );
        globalWorkSize[1] = mkGlobalDim(buffs.heightA, TILE );
    } else { // treat the case for the register + block tiling.
        kernel = kers.rgblkMMM;
        localWorkSize [0] = RT;
        localWorkSize [1] = RT;
        globalWorkSize[0] = ( (buffs.widthB + RT*RT - 1) / (RT*RT) ) * RT;
        globalWorkSize[1] = mkGlobalDim(buffs.heightA, RT );
    }

    { // run kernel
        cl_int ciErr1 = CL_SUCCESS;

        // make two dry runs
        for (int32_t i=0; i<2; i++) { 
            ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kernel, 2, NULL,
                                             globalWorkSize, localWorkSize, 0, NULL, NULL);
            clFinish(ctrl.queue);
            OPENCL_SUCCEED(ciErr1);
        }

        int64_t elapsed, aft, bef = get_wall_time();
        { // timing runs
            for (int32_t i = 0; i < RUNS_GPU; i++) {
                ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kernel, 2, NULL,
                                                 globalWorkSize, localWorkSize, 0, NULL, NULL);
            }
            clFinish(ctrl.queue);
        }
        aft = get_wall_time();
        elapsed = aft - bef;
        OPENCL_SUCCEED(ciErr1);
        {
            double microsecPerMatrixMul = elapsed/RUNS_GPU; 
            double flopsPerMatrixMul = 2.0 * buffs.heightA * buffs.widthB * buffs.widthA; 
            double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f));
            if      (kind == NAIVE) printf("GPU Naive ");
            else if (kind == BLOCK) printf("GPU Block-Tiled ");
            else if (kind == RGBLK) printf("GPU Register + Block Tiled ");
            printf("dense matrix-matrix multiplication average runtime: %dμs (across %d runs), GFLOPs: %.2f ... ",
                   (uint32_t)(elapsed/RUNS_GPU), RUNS_GPU, gigaFlops);
        }
    }

    { // transfer result to CPU and validate
        const uint32_t N = buffs.heightA * buffs.widthB;
        gpuToCpuTransfer(N, hdC);
        validate(hC, hdC, N);
        memset(hdC, 0, N*sizeof(real));
    }
} 

/*******************************/
/***     RUN ALL VERSIONS    ***/
/*******************************/

void testForSizes( const uint32_t heightA
                 , const uint32_t widthB
                 , const uint32_t widthA
) {
    // allocate and init host arrays
    real* hA = (real*) malloc( heightA * widthA * sizeof(real) );
    real* hB = (real*) malloc( widthA  * widthB * sizeof(real) );
    real* hC = (real*) malloc( heightA * widthB * sizeof(real) );
    real* hdC= (real*) malloc( heightA * widthB * sizeof(real) );

    randomInit(hA, heightA * widthA);
    randomInit(hB, widthA  * widthB);

    // run golden sequential version
    goldMMM(hA, hB, hC, heightA, widthB, widthA);

    // initialize opencl control parameters, buffers and kernels.
    initOclBuffers ( heightA, widthB, widthA, hA, hB );
    initKernels();

    // run the gpu versions
    runGPUverMMM(NAIVE, hC, hdC);
    runGPUverMMM(BLOCK, hC, hdC);
    runGPUverMMM(RGBLK, hC, hdC);

    // free Ocl Kernels and Buffers
    freeOclBuffKers();
    
    // finally free the host-allocated buffers
    free(hA); free(hB); free(hC); free(hdC);
}

int main() {
    initOclControl();
    testForSizes(HEIGHT_A, WIDTH_B, WIDTH_A);
    freeOclControl();
}
