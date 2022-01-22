#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 

#include "mmm-kernels.cu.h"

#define GPU_RUNS    100
#define SMALL       0
#define LONGINNER   0

#if SMALL

    #define WIDTH_A  64
    #define HEIGHT_A 10
    #define WIDTH_B  500
    #define TILE     16

    #define Ty  10
    #define Tx  16
    #define Ry  1
    #define Rx  1
    #define Tk  16
    #define Rk  1

#elif LONGINNER

    #define WIDTH_A  (1024*1024)//1024 //1024//2048
    #define HEIGHT_A 64//2048//2048//2048
    #define WIDTH_B  64//2048
    #define TILE     16

    #define Ty  16
    #define Tx  16
    #define Ry  4
    #define Rx  4
    #define Tk  32
    #define Rk  1024

#else

#if 1
    #define WIDTH_A  4096//1024 //(1024+17)//1024 //1024//2048
    #define HEIGHT_A 2048//1024 //(1024+19)//2048//2048//2048
    #define WIDTH_B  2048//1024 //(1024+23)//4096//2048
    #define TILE     16//16

    #define Ty  16
    #define Tx  16
    #define Ry  4
    #define Rx  4
    #define Tk  16
    #define Rk  8 //32
#else
    #define WIDTH_A  (1024+17)//1024 //1024//2048
    #define HEIGHT_A (1024+19)//2048//2048//2048
    #define WIDTH_B  (1024+23)//4096//2048
    #define TILE     16//16

    #define Ty  16
    #define Tx  27
    #define Ry  5
    #define Rx  5
    #define Tk  19
    #define Rk  32
#endif

#endif 

/////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////

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
void matMult(T* A, T* B, T* C, int colsA, int rowsA, int colsB) {
  for(int i = 0; i < rowsA; i++) {
    for(int j = 0; j < colsB; j++) {
      float sum = 0.0;
      for(int k = 0; k < colsA; k++) {
        sum += A[i*colsA + k] * B[k * colsB + j];
      }
      C[i * colsB + j] = sum;
    }
  } 
}

template<class T>
bool validate(float* A,float* B, unsigned int sizeAB){
    for(int i = 0; i < sizeAB; i++)
      if (fabs(A[i] - B[i]) > 0.02) { //0.0007){
        printf("INVALID RESULT %d %f %f\n", i, A[i], B[i]);
        return false;
      }
    printf("VALID RESULT!\n");
    return true;
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main() {
   // set seed for rand()
   srand(2006);
 
   // 1. allocate host memory for the two matrices
   unsigned int size_A = WIDTH_A * HEIGHT_A;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* h_A = (float*) malloc(mem_size_A);
 
   unsigned int size_B = WIDTH_B * WIDTH_A;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* h_B = (float*) malloc(mem_size_B);
 
   // 2. initialize host memory
   randomInit(h_A, size_A);
   randomInit(h_B, size_B);
    
   // 3. allocate device memory
   float* d_A;
   float* d_B;
   cudaMalloc((void**) &d_A, mem_size_A);
   cudaMalloc((void**) &d_B, mem_size_B);
 
   // 4. copy host memory to device
   cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
 
   // 5. allocate host memory for the result C
   unsigned int size_C = HEIGHT_A * WIDTH_B;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* h_C   = (float*) malloc(mem_size_C);
   float* seq_C = (float*) malloc(mem_size_C);
 
   // 6. allocate device memory for the result
   float *d_C;
   cudaMalloc((void**) &d_C, mem_size_C);
 
   printf("Sizes are: (HeightA, WidthB, WidthA)=(%d, %d, %d)\n", HEIGHT_A, WIDTH_B, WIDTH_A);

   // 7. compute sequential matrix multiplication
   {
      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      matMult<float>(h_A, h_B, seq_C, WIDTH_A, HEIGHT_A, WIDTH_B);

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
      printf("Sequential Naive version runs in: %lu microsecs\n", elapsed);
   }

   

   // execute the naive kernel
   {
      // setup execution parameters
      int  dimy = ceil( ((float)HEIGHT_A)/TILE ); 
      int  dimx = ceil( ((float) WIDTH_B)/TILE );
      dim3 block(TILE, TILE, 1);
      dim3 grid (dimx, dimy, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      matMultKer<float> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A);
      cudaThreadSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);

      // copy result from device to host
      cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
      // validate
      printf("GPU Naive MMM version ... ");
      validate<float>(seq_C, h_C, size_C);

      printf("GPU Naive MMM version runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed; 
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f)); 
      printf( "GPU Naive MMM Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y); 
   }

 
   // execute the block-tiled kernel
   {
      // setup execution parameters
      int  dimy = ceil( ((float)HEIGHT_A)/TILE ); 
      int  dimx = ceil( ((float) WIDTH_B)/TILE );
      dim3 block(TILE, TILE, 1);
      dim3 grid (dimx, dimy, 1);

      { //dry run
        matMultTiledKer<float,TILE> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A); 
        //matMultCacheKer<float,TILE> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A);
        cudaThreadSynchronize();
      }

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      for(int i=0; i<GPU_RUNS; i++) {
          matMultTiledKer<float,TILE> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A); 
      }
      cudaThreadSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

      // copy result from device to host
      cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
      // validate
      printf("GPU Block-Tiled MMM version ... ");
      validate<float>(seq_C, h_C, size_C);

      printf("GPU Block-Tiled MMM version runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed; 
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f)); 
      printf( "GPU Block-Tiled MMM Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y); 
   }

   // execute the block+register tiled kernel
   // ToDo: please fill in the implementation below
   //       specialized for TILE = 16
   {
      // setup execution parameters
      int  dimy = ceil( ((float)HEIGHT_A)/TILE ); 
      int  dimx = ceil( ((float) WIDTH_B)/(TILE*TILE) );
      dim3 block(TILE, TILE, 1);
      dim3 grid (dimx, dimy, 1);

      //printf("\nDimy: %d, dimx: %d, tile: %d\n", dimy, dimx, TILE);

      { // one dry run
        matMultRegTiledKer<float,TILE> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A); 
        cudaThreadSynchronize();
      }
      cudaMemset(d_C, 0, mem_size_C);
      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      for(int i=0; i<GPU_RUNS; i++) {
        matMultRegTiledKer<float,TILE> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A); 
      }
      cudaThreadSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

      // copy result from device to host
      cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
      // validate
      printf("GPU Block+Register Tiled MMM version ... ");
      validate<float>(seq_C, h_C, size_C);

      printf("GPU Block+Register Tiled MMM version runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed; 
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f)); 
      printf( "GPU Block+Register Tiled MMM Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y); 
   }


   // execute the block+register tiled(Ty,Ry,Tx,Rx,Tk) kernel, with both dimensions tiled!
   {
      // setup execution parameters
      int  dimy = ceil( ((float)HEIGHT_A)/(Ty*Ry) ); 
      int  dimx = ceil( ((float) WIDTH_B)/(Tx*Rx) );
      dim3 block(Tx, Ty, 1);
      dim3 grid (dimx, dimy, 1);

      { // one dry run
        mmmTnRn<float,Ty,Ry,Tx,Rx,Tk> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A); 
        cudaThreadSynchronize();
      }

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      for(int i=0; i<GPU_RUNS; i++) {
        mmmTnRn<float,Ty,Ry,Tx,Rx,Tk> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A);
      }
      cudaThreadSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

      // copy result from device to host
      cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
      // validate
      printf("GPU B+R Tiled(Ty,Ry,Tx,Rx,Tk) MMM version ... ");
      validate<float>(seq_C, h_C, size_C);

      printf("GPU B+R Tiled(Ty,Ry,Tx,Rx,Tk) MMM version runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed; 
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f)); 
      printf( "GPU B+R Tiled(Ty,Ry,Tx,Rx,Tk) MMM Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y); 
   }

   // exploits parallelism from all dimensions, including REDoMAP;
   // (Ty,Ry,Tx,Rx,Tk) are as before
   // Rk is not a register tile size per say, it just means that
   //    a factor Rk*Tk of the WIDTH_A dimension is going to be
   //    sequentialized, and (WIDTH_A / (Rk*Tk)) is going to be
   //    parallelized
   {
      // setup execution parameters
      int  dimy = ceil( ((float)HEIGHT_A)/(Ty*Ry) ); 
      int  dimx = ceil( ((float) WIDTH_B)/(Tx*Rx) );
      int  dimz = ceil( ((float) WIDTH_A)/(Tk*Rk) );
      dim3 block(Tx, Ty, 1);
      dim3 grid (dimx, dimy, dimz);

      float *d_Cext;
      cudaMalloc((void**) &d_Cext, mem_size_C*dimz);
    
      const unsigned int blockred = 256; 
      const unsigned int dimred = (HEIGHT_A*WIDTH_B + blockred - 1) / blockred;
      cudaMemset(d_C, 0, mem_size_C);

      { // one dry run
        cudaMemset(d_C, 0, mem_size_C);
        mmmTnRnPar<float,Ty,Ry,Tx,Rx,Tk, Rk> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A);
//        mmmTnRnPar<float,Ty,Ry,Tx,Rx,Tk, Rk> <<< grid, block >>>(d_A, d_B, d_Cext, HEIGHT_A, WIDTH_B, WIDTH_A);
//        seqRedInner<float> <<<dimred, blockred>>>(d_Cext, d_C, HEIGHT_A*WIDTH_B, dimz); 
        cudaThreadSynchronize();
      }

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      for(int i=0; i<GPU_RUNS; i++) {
        cudaMemset(d_C, 0, mem_size_C);
        mmmTnRnPar<float,Ty,Ry,Tx,Rx,Tk, Rk> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A);
//        mmmTnRnPar<float,Ty,Ry,Tx,Rx,Tk,Rk> <<< grid, block >>>(d_A, d_B, d_Cext, HEIGHT_A, WIDTH_B, WIDTH_A);
//        seqRedInner<float> <<<dimred, blockred>>>(d_Cext, d_C, HEIGHT_A*WIDTH_B, dimz); 
      }
      cudaThreadSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

      // copy result from device to host
      cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
      // validate
      printf("GPU B+R Tiled(Ty,Ry,Tx,Rx,Tk) Par-Inner MMM version ... ");
      validate<float>(seq_C, h_C, size_C);

      printf("GPU B+R Tiled(Ty,Ry,Tx,Rx,Tk) Par-Inner MMM version runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed; 
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f)); 
      printf( "GPU B+R Tiled(Ty,Ry,Tx,Rx,Tk) Par-Inner MMM Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y); 
   }


   // 7. clean up memory
   free(h_A);
   free(h_B);
   free(h_C);
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
}

