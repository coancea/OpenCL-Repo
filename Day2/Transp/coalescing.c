#include "../../clutils.h"
#include <stdbool.h>
#include <math.h>
#include "bridge.h"

#define TILE   16
#define CHUNK  8
#define HEIGHT 65536 //537
#define WIDTH  128   //422

#define RUNS_CPU 1
#define RUNS_GPU 50

#include "helper.h"
/*******************************/
/***        Helpers          ***/
/*******************************/

void randomInit(real* data, const uint32_t size) {
    for (uint32_t i = 0; i < size; ++i)
        data[i] = rand() / (real)RAND_MAX;
}

void goldTransp(real* A, real* B, uint32_t height, uint32_t width) {
  int64_t bef = get_wall_time();

  for(int l = 0; l < RUNS_CPU; l++) {
    for(uint32_t i = 0; i < height; i++) {
      for(uint32_t j = 0; j < width; j++) {
        B[j*height + i] = A[i*width + j];
      }
    } 
  }

  int64_t aft = get_wall_time();
  int64_t elapsed_us = aft-bef;

  printf("Golden (sequential) transposition average runtime: %dμs (across %d runs)\n",
         (uint32_t)(elapsed_us/RUNS_CPU), RUNS_CPU);
}

bool validate(real* A, real* B, uint32_t sizeAB){
    for(uint32_t i = 0; i < sizeAB; i++)
      if (fabs(A[i] - B[i]) > 0.00001){
        printf("INVALID RESULT %d %f %f\n", i, A[i], B[i]);
        return false;
      }
    printf("VALID RESULT!\n");
    return true;
}

inline size_t mkGlobalDim(const uint32_t pardim, const uint32_t T) {
    return ((pardim + T - 1) / T) * T;
}

void runGPUverTransp(TranspVers kind, real* hB, real* hdB) {
    size_t localWorkSize[2];
    size_t globalWorkSize[2];
    cl_kernel kernel;

    if ( (kind == NAIVE_TRANSP) || (kind == COALS_TRANSP) ) {
        kernel = (kind == NAIVE_TRANSP) ? kers.naiveTransp : kers.coalsTransp;
        localWorkSize [0] = TILE;
        localWorkSize [1] = TILE;
        globalWorkSize[0] = mkGlobalDim(buffs.width , TILE );
        globalWorkSize[1] = mkGlobalDim(buffs.height, TILE );
    } else { // treat the case for coalesced + chunked transposition
        kernel = kers.optimTransp;
        localWorkSize [0] = TILE;
        localWorkSize [1] = TILE;
        globalWorkSize[0] = mkGlobalDim(buffs.width, TILE);
        globalWorkSize[1] = ( (buffs.height + TILE*CHUNK - 1) / (TILE*CHUNK) ) * TILE;
    }

    { // run kernel
        cl_int ciErr1 = CL_SUCCESS;

        // make two dry runs
        for (int32_t i=0; i<1; i++) { 
            ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kernel, 2, NULL,
                                             globalWorkSize, localWorkSize, 0, NULL, NULL);
            clFinish(ctrl.queue);
            OPENCL_SUCCEED(ciErr1);
        }

        int64_t elapsed, aft, bef = get_wall_time();
        { // timing runs
#if 1
            for (int32_t i = 0; i < RUNS_GPU; i++) {
                ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kernel, 2, NULL,
                                                 globalWorkSize, localWorkSize, 0, NULL, NULL);
            }
            clFinish(ctrl.queue);
#endif
        }
        aft = get_wall_time();
        elapsed = aft - bef;
        OPENCL_SUCCEED(ciErr1);
        {
            double microsecPerTransp = elapsed/RUNS_GPU; 
            double bytesaccessed = 2 * buffs.height * buffs.width * sizeof(real); // one read + one write
            double gigaBytesPerSec = (bytesaccessed * 1.0e-9f) / (microsecPerTransp / (1000.0f * 1000.0f));
            if      (kind == NAIVE_TRANSP) printf("GPU Naive ");
            else if (kind == COALS_TRANSP) printf("GPU Coalesced ");
            else if (kind == OPTIM_TRANSP) printf("GPU Coalesced + Chunked ");
            printf("TRANSPOSITION average runtime: %dμs (across %d runs), Gbytes/sec: %.2f ... ",
                   (uint32_t)(elapsed/RUNS_GPU), RUNS_GPU, gigaBytesPerSec);
        }
    }

    { // transfer result to CPU and validate
        const uint32_t N = buffs.height * buffs.width;
        gpuToCpuTransfer(N, hdB);
        validate(hB, hdB, N);
        memset(hdB, 0, N*sizeof(real));
    }
} 

/*******************************/
/***     RUN ALL VERSIONS    ***/
/*******************************/

void testForSizes( const uint32_t height
                 , const uint32_t width
) {
    // allocate and init host arrays
    size_t memsize = height * width * sizeof(real);
    real* hA = (real*) malloc( memsize );
    real* hB = (real*) malloc( memsize );
    real* hdB= (real*) malloc( memsize );

    randomInit(hA, height * width);

    // run golden sequential version
    goldTransp(hA, hB, height, width);

    // initialize opencl control parameters, buffers and kernels.
    initOclBuffers ( height, width, hA );
    initTranspKernels();

    // run the gpu versions of transpose
    runGPUverTransp(NAIVE_TRANSP, hB, hdB);
    runGPUverTransp(COALS_TRANSP, hB, hdB);
    runGPUverTransp(OPTIM_TRANSP, hB, hdB);

    // free Ocl Kernels and Buffers
    freeOclBuffKers();
    
    // finally free the host-allocated buffers
    free(hA); free(hB); free(hdB);
}

int main() {
    initOclControl();
    testForSizes(HEIGHT, WIDTH);
    freeOclControl();
}
