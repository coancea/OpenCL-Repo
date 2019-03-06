#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 

#include "transpose-kernels.cu.h"
#include "transpose-host.cu.h"

#define HEIGHT_A 1024*8   //12835//2048//2048
#define  WIDTH_A 1024*8  //15953 //1024//2048
#define TILE     32

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}


void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}


template<class T>
void matTranspose(T* A, T* trA, int rowsA, int colsA) {
  for(int i = 0; i < rowsA; i++) {
    for(int j = 0; j < colsA; j++) {
      trA[j*rowsA + i] = A[i*colsA + j];
    }
  }
}

template<class T>
bool validateTranspose(float* A,float* trA, unsigned int rowsA, unsigned int colsA){
  bool valid = true;
  for(unsigned int i = 0; i < rowsA; i++) {
    for(unsigned int j = 0; j < colsA; j++) {
      if(trA[j*rowsA + i] != A[i*colsA + j]) {
        printf("row: %d, col: %d, A: %.4f, trA: %.4f\n", 
                i, j, A[i*colsA + j], trA[j*rowsA + i] );
        valid = false;
        break;
      }
    }
    if(!valid) break;
  }
  if (valid) printf("GPU TRANSPOSITION   VALID!\n");
  else       printf("GPU TRANSPOSITION INVALID!\n");
  return valid;
}


bool validateProgram(float* A, float* B, unsigned int N){
  bool valid = true;
  for(unsigned int i = 0; i < N; i++) {
    unsigned long long ii = i*64;
    float tmpB = A[i*64];
    tmpB = tmpB*tmpB;
    if(fabs(B[ii] - tmpB)> 0.00001) { valid = false; break; }
    for(int j = 1; j < 64; j++) {
        float tmpA  = A[ii + j];
        float accum = sqrt(tmpB) + tmpA*tmpA;

        if(fabs(B[ii+j] - accum) > 0.00001) { valid = false; break; }
        tmpB        = accum;
    }
    if(!valid) break;
  }
  if (valid) printf("GPU PROGRAM   VALID!\n");
  else       printf("GPU PROGRAM INVALID!\n");
  return valid;
}


int main() {
    // set seed for rand()
    srand(2006);
 
    // 1. allocate host memory for the two matrices
    size_t size_A = WIDTH_A * HEIGHT_A;
    size_t mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    float* h_B = (float*) malloc(mem_size_A);
 
    // 2. initialize host memory
    randomInit(h_A, size_A);
    
    // 3. allocate device memory
    float* d_A;
    float* d_B;
    cudaMalloc((void**) &d_A, mem_size_A);
    cudaMalloc((void**) &d_B, mem_size_A);
 
    // 4. copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    { // test transpose
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        //transposeNaive<float, TILE>( d_A, d_B, HEIGHT_A, WIDTH_A );
        transposeTiled<float, TILE>( d_A, d_B, HEIGHT_A, WIDTH_A );

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("Transpose on GPU runs in: %lu microsecs\n", elapsed);

        // copy result from device to host
        cudaMemcpy(h_B, d_B, mem_size_A, cudaMemcpyDeviceToHost);
  
        // validate
        validateTranspose<float>( h_A, h_B, HEIGHT_A, WIDTH_A );
    }

    const unsigned int REPEAT = 32;
    { // compute original program
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        unsigned int num_thds= (HEIGHT_A/64)*WIDTH_A;
        unsigned int block   = 256;
        unsigned int grid    = num_thds / block;

        for (int kkk = 0; kkk < REPEAT; kkk++) {
            origProg<<<grid,block>>>(d_A, d_B, num_thds);
        }
        cudaThreadSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("Original Program on GPU runs in: %lu microsecs\n", elapsed/REPEAT);

        // copy result from device to host
        cudaMemcpy(h_B, d_B, mem_size_A, cudaMemcpyDeviceToHost);

        validateProgram(h_A, h_B, num_thds);
    }

    { // compute transformed program

        float* d_Atr;   cudaMalloc((void**) &d_Atr, mem_size_A);
        float* d_Btr;   cudaMalloc((void**) &d_Btr, mem_size_A);

        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        unsigned int num_thds= (HEIGHT_A/64)*WIDTH_A;
        unsigned int block   = 256;
        unsigned int grid    = num_thds / block;

        for (int kkk = 0; kkk < REPEAT; kkk++) {
            transposeTiled<float, TILE>( d_A, d_Atr, num_thds, 64 );
            transfProg<<<grid,block>>>(d_Atr, d_Btr, num_thds);
            transposeTiled<float, TILE>( d_Btr, d_B, 64, num_thds );
        }
        cudaThreadSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("Transformed Program on GPU runs in: %lu microsecs\n", elapsed/REPEAT);

        // copy result from device to host
        cudaMemcpy(h_B, d_B, mem_size_A, cudaMemcpyDeviceToHost);
        validateProgram(h_A, h_B, num_thds);

        cudaFree(d_Atr);
        cudaFree(d_Btr);
   }

   // clean up memory
   free(h_A);
   free(h_B);
   cudaFree(d_A);
   cudaFree(d_B);
}

