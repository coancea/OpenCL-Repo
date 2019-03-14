#ifndef TENSOR_KERNELS
#define TENSOR_KERNELS

#define A4(X,len,I1,I2,I3,I4) (X[ (I1)*(len)*(len)*(len) + \
                                  (I2)*(len)*(len) + \
                                  (I3)*(len) + (I4) ])

#define A6(X,len,I1,I2,I3,I4,I5,I6) (X[ (I1)*(len)*(len)*(len)*(len)*(len) + \
                                        (I2)*(len)*(len)*(len)*(len) + \
                                        (I3)*(len)*(len)*(len) + \
                                        (I4)*(len)*(len) + (I5)*(len) + (I6) ])

template <class ElTp, int T> 
__global__ void tensorProdNaiveKer(ElTp* A, ElTp* B, ElTp* C, const int len) {
  int i, ii, j, jj, aa, k, kk, c, cc, bb;
  int tmp;

  { // compute i, j, a, k, c, b
    j = threadIdx.y / T;
    i = threadIdx.y % T;

    k = threadIdx.x / T;
    c = threadIdx.x % T;
  }

  { // compute ii, jj, aa, kk, cc, bb
    int num_d = (len + T - 1) / T;
    jj = (blockIdx.y / (num_d*num_d)) * T;
    tmp = blockIdx.y % (num_d*num_d);
    ii = (tmp / num_d) * T;
    aa = (tmp % num_d) * T;

    kk = (blockIdx.x / (num_d*num_d)) * T;
    tmp = blockIdx.x % (num_d*num_d);
    cc = (tmp / num_d) * T;
    bb = (tmp % num_d) * T;
  }

  if ( (j+jj >= len) || (i+ii >= len) || (aa >= len) ||
       (k+kk >= len) || (c+cc >= len) || (bb >= len)  )
    return; // out of range

  for(int a=0; a<T; a++) {
    for(int b=0; b<T; b++) {
      ElTp accum = 0.0;
      for(int d=0; d<len; d++) {
        ElTp x = A4(A,len,jj+j,ii+i,aa+a,d);
        ElTp y = A4(B,len,kk+k,cc+c,bb+b,d);
        accum +=  x * y;
      }
      A6(C,len,kk+k,jj+j,ii+i,cc+c,bb+b,aa+a) = accum;
    }
  }
}

#endif // TENSOR_KERNELS
