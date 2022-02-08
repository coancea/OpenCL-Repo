#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 
#include <cstdlib>

#include <curand.h>
#include <cublas_v2.h>

// nvcc main-cublas.cu -lcublas -lcurand

#define GPU_RUNS 50

#define HEIGHT_A 2048//1024 //(1024+19)//2048//2048//2048
#define WIDTH_A  4096//1024 //(1024+17)//1024 //1024//2048
#define WIDTH_B  2048//1024 //(1024+23)//4096//2048

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
     int lda=m,ldb=k,ldc=m;
     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;

     // Do the actual multiplication
     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
 
// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul_rep(const float *A, const float *B, float *C, const int m, const int k, const int n) {
     int lda=m,ldb=k,ldc=m;
     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;
 
     // Create a handle for CUBLAS
     cublasHandle_t handle;
     cublasCreate(&handle);
 
     printf("AAAAA\n");

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
     
     // Do the actual multiplication
      //for(int i=0; i < GPU_RUNS; i++) {
         cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
      //}
      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 

    printf("before BBBB\n");

      float microsecPerMatrixMul = elapsed; 
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f)); 

      printf("CUBLAS version runs in: %lu microsecs, GFlops/sec: %f\n", elapsed/GPU_RUNS, gigaFlops);

     // Destroy the handle
     cublasDestroy(handle);
}

int main() {
     // Allocate 3 arrays on CPU
     int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
 
     // for simplicity we are going to use square arrays
     nr_rows_A = HEIGHT_A;
     nr_cols_A = WIDTH_A;
     nr_rows_B = nr_cols_A;
     nr_cols_B = WIDTH_B;
     nr_rows_C = nr_rows_A;
     nr_cols_C = nr_cols_B;
 
     float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
     float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
     float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
 
     // Allocate 3 arrays on GPU
     float *d_A, *d_B, *d_C;
     cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
     cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
     cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));
 
    printf("before MMM 111\n");

     // Fill the arrays A and B on GPU with random numbers
     GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
     GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
 
     // Optionally we can copy the data back on CPU and print the arrays
     cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
     cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
//     std::cout << "A =" << std::endl;
//     print_matrix(h_A, nr_rows_A, nr_cols_A);
//     std::cout << "B =" << std::endl;
//     print_matrix(h_B, nr_rows_B, nr_cols_B);
 
     printf("before MMM\n");

     // Multiply A and B on GPU
     gpu_blas_mmul_rep(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
 
     printf("after MMM\n");

     // Copy (and print) the result on host memory
     cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
//     std::cout << "C =" << std::endl;
//     print_matrix(h_C, nr_rows_C, nr_cols_C);
 
     //Free GPU memory
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
 
     // Free CPU memory
     free(h_A);
     free(h_B);
     free(h_C);
 
     return 0;
}
