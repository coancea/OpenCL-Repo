#ifndef MULT_KERNELS
#define MULT_KERNELS

// widthA = heightB
template <class ElTp> 
__global__ void matMultKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  if( (gidx >= widthB) || (gidy >= heightA) ) return;

  for(int k = 0; k < widthA; k ++) {
      accum += A[gidy*widthA + k] * B[k*widthB + gidx];
  }

  C[gidy*widthB + gidx] = accum;
}


// widthA = heightB
template <class ElTp, int T> 
__global__ void matMultTiledKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  __shared__ ElTp Ash[T][T];
  __shared__ ElTp Bsh[T][T];

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  for(int kk = 0; kk < widthA; kk += T) {
      Ash[threadIdx.y][threadIdx.x] = ((gidy < heightA) && (kk+threadIdx.x < widthA)) ?
            A[gidy*widthA + kk + threadIdx.x] : 0.0;
      Bsh[threadIdx.y][threadIdx.x] = ((gidx < widthB)  && (kk+threadIdx.y < widthA)) ?
            B[(threadIdx.y+kk)*widthB + gidx] : 0.0;
      __syncthreads();

      #pragma unroll
      for(int k = 0; k < T; k++)
          accum += Ash[threadIdx.y][k] * Bsh[k][threadIdx.x];
      __syncthreads();
  }

  if( (gidx < widthB) && (gidy < heightA) )
    C[gidy*widthB + gidx] = accum;
}

// widthA = heightB
template <class ElTp, int T> 
__global__ void matMultCacheKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  for(int kk = 0; kk < widthA; kk += T) {
      __syncthreads();
      #pragma unroll
      for(int k = 0; k < T; k++)
        accum += A[gidy*widthA + kk + k] * B[gidy*widthB + (kk+k)];
  }

  if( (gidx < widthB) && (gidy < heightA) )
    C[gidy*widthB + gidx] = accum;
}


template <class ElTp, int T> 
__global__ void matMultRegTiledKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  __shared__ ElTp Ashreg[T][T];
  ElTp cs[T];

  unsigned int ii  = blockIdx.y * T;
  unsigned int jjj = blockIdx.x * T * T;
  unsigned int jj  = jjj + threadIdx.y * T;
  unsigned int j   = jj  + threadIdx.x;

  #pragma unroll
  for(int i=0; i<T; i++)
    cs[i] = 0.0;

  for(int kk = 0; kk < widthA; kk += T) {
      ElTp tmp = 0;
      if ((ii+threadIdx.y < heightA) && (kk+threadIdx.x < widthA)) {
        tmp = A[(ii+threadIdx.y)*widthA + kk+threadIdx.x];
      }
      Ashreg[threadIdx.y][threadIdx.x] = tmp;
      __syncthreads();

      for(int k = 0; k < T; k++) {
          ElTp b = 0;
          if ((k+kk < widthA) && (j < widthB)) {
            b = B[(k+kk)*widthB + j];
          }

          #pragma unroll 
          for(int i=0; i<T; i++) {
            cs[i] += Ashreg[i][k] * b;
          }
      }
      __syncthreads();
  }

  #pragma unroll
  for(int i=0; i<T; i++) {
    if( (ii+i < heightA) && (j < widthB) )
      C[(ii+i)*widthB + j] = cs[i];
  }
}

/************************************************/
/*** Block+Register Tile with different tiles ***/
/*** the parallel dimensions and the seq one  ***/
/************************************************/

template <class ElTp, int Ty, int Ry, int Tx, int Rx, int Tk>
__global__ void mmmTnRn(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  __shared__ ElTp Aloc[Ty*Ry][Tk];
  __shared__ ElTp Bloc[Tk][Tx*Rx]; 
  ElTp css[Ry][Rx];
  ElTp as[Ry];
  ElTp bs[Rx];

  unsigned int iii = blockIdx.y * Ty * Ry;
  unsigned int jjj = blockIdx.x * Tx * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++)
    #pragma unroll
    for(int j=0; j<Rx; j++)
      css[i][j] = 0.0;

  for(int kk = 0; kk < widthA; kk += Tk) {

      // copy the slice of A: Ashreg = A[iii : iii + Ty*Ry , kk : kk+Tk]
      //   such that the accesses to A and Aloc are both coalesced!
      for(int i = threadIdx.y; i < Ty*Ry; i+=Ty) {
          for(int k = threadIdx.x; k < Tk; k+=Tx) {
              ElTp v = 0.0;
              if ( (iii+i < heightA) && (kk+k < widthA) )
                  v = A[(iii+i)*widthA + (kk+k)];
              Aloc[i][k] = v;
          }
      }

      // copy the slice of B: Bshreg = B[kk : kk+Tk , jjj : jjj + Tx*Rx]
      //   such that the accesses to B and Bloc are both coalesced!
      for(int k = threadIdx.y; k < Tk; k+=Ty) {
          for(int j = threadIdx.x; j < Tx*Rx; j+=Tx) {
              ElTp v = 0.0;
              if ( (jjj+j < widthB) && (kk+k < widthA) )
                  v = B[(kk+k)*widthB + (jjj + j)];
              Bloc[k][j] = v;
          }
      }
      __syncthreads();

      for(int k = 0; k < Tk; k++) {
          // copy from local to register memory for A
          #pragma unroll
          for(int i = 0; i < Ry; i++) {
              as[i] = Aloc[threadIdx.y*Ry+i][k];
          }
          // copy from local to register memory for B
          #pragma unroll
          for(int j = 0; j < Rx; j++) {
              bs[j] = Bloc[k][threadIdx.x*Rx+j];
          }

          #pragma unroll
          for(int i=0; i<Ry; i++) {
            #pragma unroll
            for(int j=0; j<Rx; j++) {
                // unfortunately we need a safety condition here
                // or do we? because if i or j is out of range then
                // cs[i][j] is invalid anyways -- so everything looks safe!
                css[i][j] += as[i] * bs[j];
            }
          }
      }
      __syncthreads();
  }

  unsigned int indy = iii + threadIdx.y * Ry;
  unsigned int indx = jjj + threadIdx.x * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++) {
    #pragma unroll
    for(int j=0; j<Rx; j++) {
      if( (indy+i < heightA) && (indx+j < widthB) )
        C[(indy+i)*widthB + (indx+j)] = css[i][j];
    }
  }
}


template <class ElTp, int Ty, int Ry, int Tx, int Rx, int Tk>
__global__ void mmmTnRn_Play(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  __shared__ ElTp Aloc[Ty*Ry][Tk];
  __shared__ ElTp Bloc[Tk][Tx*Rx]; 
  ElTp css[Ry][Rx];
  ElTp as[Ry];
  ElTp bs[Rx];

  unsigned int iii = blockIdx.y * Ty * Ry;
  unsigned int jjj = blockIdx.x * Tx * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++)
    #pragma unroll
    for(int j=0; j<Rx; j++)
      css[i][j] = 0.0;

  for(int kk = 0; kk < widthA; kk += Tk) {

      // copy the slice of A: Ashreg = A[iii : iii + Ty*Ry , kk : kk+Tk]
      //   such that the accesses to A and Aloc are both coalesced!
      for(int i = threadIdx.y; i < Ty*Ry; i+=Ty) {
          for(int k = threadIdx.x; k < Tk; k+=Tx) {
              ElTp v = 3.0;
              if ( (iii+i < heightA) && (kk+k < widthA) )
                  v = A[(iii+i)*widthA + (kk+k)];
              Aloc[i][k] = v;
          }
      }

      // copy the slice of B: Bshreg = B[kk : kk+Tk , jjj : jjj + Tx*Rx]
      //   such that the accesses to B and Bloc are both coalesced!
      for(int k = threadIdx.y; k < Tk; k+=Ty) {
          for(int j = threadIdx.x; j < Tx*Rx; j+=Tx) {
              ElTp v = 5.0;
              if ( (jjj+j < widthB) && (kk+k < widthA) )
                  v = B[(kk+k)*widthB + (jjj + j)];
              Bloc[k][j] = v;
          }
      }
      __syncthreads();

      for(int k = 0; k < Tk; k++) {
          // copy from local to register memory for A
          #pragma unroll
          for(int i = 0; i < Ry; i++) {
              as[i] = Aloc[threadIdx.y*Ry+i][k];
          }
          // copy from local to register memory for B
          #pragma unroll
          for(int j = 0; j < Rx; j++) {
              bs[j] = Bloc[k][threadIdx.x*Rx+j];
          }

          #pragma unroll
          for(int i=0; i<Ry; i++) {
            #pragma unroll
            for(int j=0; j<Rx; j++) {
                ElTp ctrb = 0;
                if ( (iii+threadIdx.y*Ry+i < heightA) &&
                     (jjj+threadIdx.x*Rx+j < widthB) &&
                     (kk + k < widthA) 
                   )
                     ctrb = as[i] * bs[j];
                else ctrb = 0;
                css[i][j] += ctrb;
            }
          }
      }
      __syncthreads();
  }

  unsigned int indy = iii + threadIdx.y * Ry;
  unsigned int indx = jjj + threadIdx.x * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++) {
    #pragma unroll
    for(int j=0; j<Rx; j++) {
      if( (indy+i < heightA) && (indx+j < widthB) )
        C[(indy+i)*widthB + (indx+j)] = css[i][j];
    }
  }
}


template <class ElTp, int Ty, int Ry, int Tx, int Rx, int Tk>
__global__ void mmmTnRn_Proto(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  __shared__ ElTp Aloc[Ty*Ry][Tk];
  __shared__ ElTp Bloc[Tk][Tx*Rx];  
  ElTp css[Ry][Rx];
  ElTp as[Ry];
  ElTp bs[Rx];

  unsigned int iii = blockIdx.y * Ty * Ry;
  unsigned int jjj = blockIdx.x * Tx * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++)
    #pragma unroll
    for(int j=0; j<Rx; j++)
      css[i][j] = 0.0;

  for(int kk = 0; kk < widthA; kk += Tk) {

      // copy the slice of A: Ashreg = A[iii : iii + Ty*Ry , kk : kk+Tk]
      //   such that the accesses to A and Aloc are both coalesced!

      //for(int i = threadIdx.y; i < Ty*Ry; i+=Ty) {
      unsigned int Tk_div_Tx = (Tk + Tx - 1) / Tx;
      for(int i = 0; i < Ry; i++) {
          for(int k0 = 0; k0 < Tk_div_Tx; k0++) {
          //for(int k = threadIdx.x; k < Tk; k+=Tx) {
              int k = threadIdx.x + k0*Tx;
              ElTp v = 3.0;
              unsigned int my_i = iii + i*Ty + threadIdx.y;
              if ( (my_i < heightA) && (kk+k < widthA) )
                  v = A[my_i*widthA + (kk+k)];
              if (k < Tk)
                Aloc[i*Ty+threadIdx.y][k] = v;
          }
      }

      // copy the slice of B: Bshreg = B[kk : kk+Tk , jjj : jjj + Tx*Rx]
      //   such that the accesses to B and Bloc are both coalesced!
      unsigned int Tk_div_Ty = (Tk + Ty - 1) / Ty;
      //for(int k = threadIdx.y; k < Tk; k+=Ty) {
      for(int k0 = 0; k0 < Tk_div_Ty; k0++) {
          int k = k0*Ty + threadIdx.y;
          //for(int j = threadIdx.x; j < Tx*Rx; j+=Tx) {
          for(int j = 0; j < Rx; j ++) {
              int loc_j = threadIdx.x + j*Tx;
              ElTp v = 5.0;
              if ( (jjj+loc_j < widthB) && (kk+k < widthA) )
                  v = B[(kk+k)*widthB + (jjj + loc_j)];

              if( k < Tk)
                  Bloc[k][loc_j] = v;
          }
      }
      __syncthreads();

      for(int k = 0; k < Tk; k++) {
          // copy from local to register memory for A
          #pragma unroll
          for(int i = 0; i < Ry; i++) {
              as[i] = Aloc[threadIdx.y*Ry+i][k];
          }

          // copy from local to register memory for B
          #pragma unroll
          for(int j = 0; j < Rx; j++) {
              bs[j] = Bloc[k][threadIdx.x*Rx+j];
          }

          #pragma unroll
          for(int i=0; i<Ry; i++) {
            #pragma unroll
            for(int j=0; j<Rx; j++) {
                ElTp ctrb = 0;
                if ( (iii+threadIdx.y*Ry+i < heightA) &&
                     (jjj+threadIdx.x*Rx+j < widthB) &&
                     (kk + k < widthA) 
                   )
                     ctrb = as[i] * bs[j];
                else ctrb = 0;
                css[i][j] += ctrb;
            }
          }
      }
      __syncthreads();
  }

  unsigned int indy = iii + threadIdx.y * Ry;
  unsigned int indx = jjj + threadIdx.x * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++) {
    #pragma unroll
    for(int j=0; j<Rx; j++) {
      if( (indy+i < heightA) && (indx+j < widthB) )
        C[(indy+i)*widthB + (indx+j)] = css[i][j];
    }
  }
}


/************************************************/
/*** All Dims Parallelized, including Redomap ***/
/************************************************/

template <class ElTp, int Ty, int Ry, int Tx, int Rx, int Tk, int Rk>
__global__ void mmmTnRnPar( ElTp* A, ElTp* B, ElTp* C, 
                            int heightA, int widthB, int widthA ) {
  __shared__ ElTp Aloc[Ty*Ry][Tk];
  __shared__ ElTp Bloc[Tk][Tx*Rx]; 
  ElTp css[Ry][Rx];
  ElTp as[Ry];
  ElTp bs[Rx];

  unsigned int iii = blockIdx.y * Ty * Ry;
  unsigned int jjj = blockIdx.x * Tx * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++)
    #pragma unroll
    for(int j=0; j<Rx; j++)
      css[i][j] = 0.0;

  unsigned int kkk = blockIdx.z * Tk * Rk;
  for(int kk = kkk; kk < min(widthA, kkk+Tk*Rk); kk += Tk) {

      // copy the slice of A: Ashreg = A[iii : iii + Ty*Ry , kk : kk+Tk]
      //   such that the accesses to A and Aloc are both coalesced!
      for(int i = threadIdx.y; i < Ty*Ry; i+=Ty) {
          for(int k = threadIdx.x; k < Tk; k+=Tx) {
              ElTp v = 0.0;
              if ( (iii+i < heightA) && (kk+k < widthA) )
                  v = A[(iii+i)*widthA + (kk+k)];
              Aloc[i][k] = v;
          }
      }

      // copy the slice of B: Bshreg = B[kk : kk+Tk , jjj : jjj + Tx*Rx]
      //   such that the accesses to B and Bloc are both coalesced!
      for(int k = threadIdx.y; k < Tk; k+=Ty) {
          for(int j = threadIdx.x; j < Tx*Rx; j+=Tx) {
              ElTp v = 0.0;
              if ( (jjj+j < widthB) && (kk+k < widthA) )
                  v = B[(kk+k)*widthB + (jjj + j)];
              Bloc[k][j] = v;
          }
      }
      __syncthreads();

      for(int k = 0; k < Tk; k++) {
          // copy from local to register memory for A
          #pragma unroll
          for(int i = 0; i < Ry; i++) {
              as[i] = Aloc[threadIdx.y*Ry+i][k];
          }
          // copy from local to register memory for B
          #pragma unroll
          for(int j = 0; j < Rx; j++) {
              bs[j] = Bloc[k][threadIdx.x*Rx+j];
          }

          #pragma unroll
          for(int i=0; i<Ry; i++) {
            #pragma unroll
            for(int j=0; j<Rx; j++) {
                // unfortunately we need a safety condition here
                // or do we? because if i or j is out of range then
                // cs[i][j] is invalid anyways -- so everything looks safe!
                css[i][j] += as[i] * bs[j];
            }
          }
      }
      __syncthreads();
  }

  unsigned int indz = blockIdx.z * heightA * widthB;
  unsigned int indy = iii + threadIdx.y * Ry;
  unsigned int indx = jjj + threadIdx.x * Rx;

  #pragma unroll
  for(int i=0; i<Ry; i++) {
    #pragma unroll
    for(int j=0; j<Rx; j++) {
      if( (indy+i < heightA) && (indx+j < widthB) )
        C[indz + (indy+i)*widthB + (indx+j)] = css[i][j];
    }
  }
}


template <class ElTp>
__global__ void seqRedInner(ElTp* Cext, ElTp* C, int lenC, int dimZ) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < lenC) {
        ElTp acc = 0.0;
        for(int i=0, ind = gid; i<dimZ; i++, ind+=lenC) {
            acc += Cext[ind];
        }
        C[gid] = acc;
    }
}
#endif
