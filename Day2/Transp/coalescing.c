#include "../../clutils.h"
#include <stdbool.h>
#include <math.h>
#include "bridge.h"

#define TILE   16
#define CHUNK  16   // for AMD use 8; for NVIDIA 16
#define HEIGHT 67537
#define WIDTH  128

#define RUNS_CPU 1
#define RUNS_GPU 75

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

void goldProgrm(real* A, real* B, const uint32_t height, const uint32_t width) {
  int64_t bef = get_wall_time();

  for(int l = 0; l < RUNS_CPU; l++) {
    for(uint32_t i = 0; i < height; i++) {
      real     accum  = 0.0;
      uint32_t offset = i*width;

      for(uint32_t j = 0; j < width; j++) {
        real tmpA = A[offset+j];
        accum = sqrt(accum) + tmpA*tmpA;
        B[offset + j] = accum;
      }
    } 
  }

  int64_t aft = get_wall_time();
  int64_t elapsed_us = aft-bef;

  printf("Golden (sequential) program average runtime: %dμs (across %d runs)\n",
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

void runGPUnaive_or_optProgram(ProgrmVers vers, real* hB, real* hdB) {
    size_t localWorkSize  = TILE*TILE;
    size_t globalWorkSize = mkGlobalDim(buffs.height, TILE*TILE );
    cl_kernel kernel = (vers == NAIVE_PROGRM) ? kers.naiveProgrm : kers.optimProgrm;
    { // run kernel
        cl_int ciErr1 = CL_SUCCESS;
        // make two dry runs
        for (int32_t i=0; i<1; i++) {
            ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kernel, 1, NULL,
                                             &globalWorkSize, &localWorkSize, 0, NULL, NULL);
            clFinish(ctrl.queue);
            OPENCL_SUCCEED(ciErr1);
        }

        int64_t elapsed, aft, bef = get_wall_time();
        { // timing runs
            for (int32_t i = 0; i < RUNS_GPU; i++) {
                ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kernel, 1, NULL,
                                                 &globalWorkSize, &localWorkSize, 0, NULL, NULL);
            }
            clFinish(ctrl.queue);
        }
        aft = get_wall_time();
        elapsed = aft - bef;
        OPENCL_SUCCEED(ciErr1);
        {
            double microsecPerTransp = elapsed/RUNS_GPU; 
            double bytesaccessed = 2 * buffs.height * buffs.width * sizeof(real); // one read + one write
            double gigaBytesPerSec = (bytesaccessed * 1.0e-9f) / (microsecPerTransp / (1000.0f * 1000.0f));
            if      (vers == NAIVE_PROGRM) printf("GPU Naive ");
            else if (vers == OPTIM_PROGRM) printf("GPU Optim ");
            printf("Program average runtime: %dμs (across %d runs), Gbytes/sec: %.2f ... ",
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

void runGPUcoalsProgram(real* hB, real* hdB) {
    size_t localWorkSize  = TILE*TILE;
    size_t globalWorkSize = mkGlobalDim(buffs.height, TILE*TILE );
    size_t localTransp[2] = {TILE, TILE};
    size_t globalTransp1[2]= {mkGlobalDim(buffs.width, TILE), mkGlobalDim(buffs.height, TILE)};
    size_t globalTransp2[2]= {mkGlobalDim(buffs.height, TILE), mkGlobalDim(buffs.width, TILE)};

    { // run kernel
        cl_int ciErr1 = CL_SUCCESS;
        // make two dry runs
        for (int32_t i=0; i<1; i++) {
            // transpose A into Atr
            clSetKernelArg(kers.coalsTransp, 0, sizeof(cl_mem), &buffs.dA);
            clSetKernelArg(kers.coalsTransp, 1, sizeof(cl_mem), &buffs.dAtr);
            clSetKernelArg(kers.coalsTransp, 2, sizeof(cl_int), &buffs.height);
            clSetKernelArg(kers.coalsTransp, 3, sizeof(cl_int), &buffs.width);
            ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.coalsTransp, 2, NULL,
                                             globalTransp1, localTransp, 0, NULL, NULL);
            OPENCL_SUCCEED(ciErr1);
            
            // execute coalesced program
            ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.coalsProgrm, 1, NULL,
                                             &globalWorkSize, &localWorkSize, 0, NULL, NULL);
            OPENCL_SUCCEED(ciErr1);

            // transpose Btr into B
            clSetKernelArg(kers.coalsTransp, 0, sizeof(cl_mem), &buffs.dBtr);
            clSetKernelArg(kers.coalsTransp, 1, sizeof(cl_mem), &buffs.dB);
            clSetKernelArg(kers.coalsTransp, 2, sizeof(cl_int), &buffs.width);
            clSetKernelArg(kers.coalsTransp, 3, sizeof(cl_int), &buffs.height);
            ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.coalsTransp, 2, NULL,
                                             globalTransp2, localTransp, 0, NULL, NULL);
            clFinish(ctrl.queue);
            OPENCL_SUCCEED(ciErr1);
        }

        int64_t elapsed, aft, bef = get_wall_time();
        { // timing runs
            for (int32_t i = 0; i < RUNS_GPU; i++) {
                // transpose A into Atr
                clSetKernelArg(kers.coalsTransp, 0, sizeof(cl_mem), &buffs.dA);
                clSetKernelArg(kers.coalsTransp, 1, sizeof(cl_mem), &buffs.dAtr);
                clSetKernelArg(kers.coalsTransp, 2, sizeof(cl_int), &buffs.height);
                clSetKernelArg(kers.coalsTransp, 3, sizeof(cl_int), &buffs.width);
                ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.coalsTransp, 2, NULL,
                                             globalTransp1, localTransp, 0, NULL, NULL);

                // execute coalesced program
                ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.coalsProgrm, 1, NULL,
                                                 &globalWorkSize, &localWorkSize, 0, NULL, NULL);
                // transpose Btr into B
                clSetKernelArg(kers.coalsTransp, 0, sizeof(cl_mem), &buffs.dBtr);
                clSetKernelArg(kers.coalsTransp, 1, sizeof(cl_mem), &buffs.dB);
                clSetKernelArg(kers.coalsTransp, 2, sizeof(cl_int), &buffs.width);
                clSetKernelArg(kers.coalsTransp, 3, sizeof(cl_int), &buffs.height);
                ciErr1 |= clEnqueueNDRangeKernel(ctrl.queue, kers.coalsTransp, 2, NULL,
                                                 globalTransp2, localTransp, 0, NULL, NULL);
            }
            clFinish(ctrl.queue);
        }
        aft = get_wall_time();
        elapsed = aft - bef;
        OPENCL_SUCCEED(ciErr1);
        {
            double microsecPerTransp = elapsed/RUNS_GPU; 
            double bytesaccessed = 2 * buffs.height * buffs.width * sizeof(real); // one read + one write
            double gigaBytesPerSec = (bytesaccessed * 1.0e-9f) / (microsecPerTransp / (1000.0f * 1000.0f));
            printf("GPU Coalesced Program average runtime: %dμs (across %d runs), Gbytes/sec: %.2f ... ",
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

    // run golden sequential transpose version
    goldTransp(hA, hB, height, width);

    // initialize opencl control parameters, buffers and kernels.
    initOclBuffers ( height, width, hA );
    initTranspKernels();

    // run the gpu versions of transpose
    runGPUverTransp(NAIVE_TRANSP, hB, hdB);
    runGPUverTransp(COALS_TRANSP, hB, hdB);
    runGPUverTransp(OPTIM_TRANSP, hB, hdB);

    // run the sequential version of the uncoalesced program
    goldProgrm(hA, hB, height, width);

    // init program kernels
    initProgramKernels();

    // run the gpu version of the program
    runGPUnaive_or_optProgram(NAIVE_PROGRM, hB, hdB);
    runGPUcoalsProgram(hB, hdB);
    runGPUnaive_or_optProgram(OPTIM_PROGRM, hB, hdB);
    // free Ocl Kernels and Buffers
    freeOclBuffKers();
    
    // finally free the host-allocated buffers
    free(hA); free(hB); free(hdB);
}

int main() {
    initOclControl();
    testForSizes(HEIGHT, WIDTH);
    testForSizes(HEIGHT, WIDTH*2);
    freeOclControl();
}