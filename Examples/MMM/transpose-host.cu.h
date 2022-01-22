#ifndef TRANSPOSE_HOST
#define TRANSPOSE_HOST

template<class ElTp, int T>
void transposeTiled ( ElTp*              inp_d,  
                      ElTp*              out_d, 
                      const unsigned int height, 
                      const unsigned int width
) {
   // 1. setup block and grid parameters
   unsigned int sh_mem_size = T * (T+1) * sizeof(ElTp); 
   int  dimy = (height+T-1) / T; 
   int  dimx = (width +T-1) / T;
   dim3 block(T, T, 1);
   dim3 grid (dimx, dimy, 1);

   //2. execute the kernel
   matTransposeTiledKer<ElTp,T><<< grid, block, sh_mem_size >>>
                       (inp_d, out_d, height, width); 
   cudaDeviceSynchronize();
}

template<class ElTp, int T>
void transposeNaive ( ElTp*              inp_d,  
                      ElTp*              out_d, 
                      const unsigned int height, 
                      const unsigned int width
) {
   // 1. setup block and grid parameters
   int  dimy = (height+T-1) / T; 
   int  dimx = (width +T-1) / T;
   dim3 block(T, T, 1);
   dim3 grid (dimx, dimy, 1);

   //2. execute the kernel
   matTransposeKer<ElTp> <<< grid, block >>>
                          (inp_d, out_d, height, width);
   cudaDeviceSynchronize();
}

#endif // TRANSPOSE
