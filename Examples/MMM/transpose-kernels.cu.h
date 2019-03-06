#ifndef TRANSPOSE_KERS
#define TRANSPOSE_KERS

// widthA = heightB
template <class T> 
__global__ void matTransposeKer(T* A, T* B, int heightA, int widthA) {

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  if( (gidx >= widthA) || (gidy >= heightA) ) return;

  B[gidx*heightA+gidy] = A[gidy*widthA + gidx];
}

// blockDim.y = T; blockDim.x = T
// each block transposes a square T
template <class ElTp, int T> 
__global__ void matTransposeTiledKer(ElTp* A, ElTp* B, int heightA, int widthA) {
  extern __shared__ char sh_mem1[];
  volatile ElTp* tile = (volatile ElTp*)sh_mem1;
  //__shared__ float tile[T][T+1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if( x < widthA && y < heightA )
      tile[threadIdx.y*(T+1) + threadIdx.x] = A[y*widthA + x];

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x; 
  y = blockIdx.x * T + threadIdx.y;

  if( x < heightA && y < widthA )
      B[y*heightA + x] = tile[threadIdx.x*(T+1) + threadIdx.y];
}


__global__ void 
origProg(float* A, float* B, unsigned int N) {
    unsigned long long gid = (blockIdx.x * blockDim.x + threadIdx.x);
    if(gid >= N) return;

    gid *= 64;
    float tmpB = A[gid];
    tmpB = tmpB*tmpB;
    B[gid] = tmpB;
    for(int j=1; j<64; j++) {
        float tmpA  = A[gid + j];
        float accum = sqrt(tmpB) + tmpA*tmpA; //tmpB*tmpB + tmpA*tmpA;
        B[gid + j]  = accum;
        tmpB        = accum;
    }
}

__global__ void 
transfProg(float* A, float* B, unsigned int N) {
    unsigned long long gid = (blockIdx.x * blockDim.x + threadIdx.x);
    if(gid >= N) return;

    float tmpB = A[gid];
    tmpB = tmpB*tmpB;
    B[gid] = tmpB;
    gid += N;
    for(int j=1; j<64; j++,gid+=N) {
        float tmpA  = A[gid];
        float accum = sqrt(tmpB) + tmpA*tmpA;
        B[gid]  = accum;
        tmpB    = accum;
    }
}

#endif
