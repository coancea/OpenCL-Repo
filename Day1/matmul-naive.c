// Simple un-tiled naively implemented matrix multiplication.  It
// does, however, use a trick to be generic in the type of elements
// multiplied.

#include "../clutils.h"

typedef float elem_t;

int main(int argc, char** argv) {
  cl_int n = 1000, m = 1000, k = 1000;
  if (argc > 1) {
    n = atoi(argv[1]);
  }
  if (argc > 2) {
    m = atoi(argv[2]);
  }
  if (argc > 3) {
    k = atoi(argv[3]);
  }

  printf("Multiplying %d x %d matrix by %d x %d\n", n, m, m, k);

  cl_context ctx;
  cl_command_queue queue;
  cl_device_id device;
  cl_int error = CL_SUCCESS;

  opencl_init_command_queue_default(&device, &ctx, &queue);

  cl_program program = opencl_build_program(ctx, device, "kernels/matmul-naive.cl", "-Delem_t=float");
  cl_kernel matmul_k = clCreateKernel(program, "matmul", &error);

  cl_mem out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n*k*sizeof(elem_t), NULL, &error);
  OPENCL_SUCCEED(error);

  cl_mem xss = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n*m*sizeof(elem_t), NULL, &error);
  OPENCL_SUCCEED(error);

  cl_mem yss = clCreateBuffer(ctx, CL_MEM_READ_ONLY, m*k*sizeof(elem_t), NULL, &error);
  OPENCL_SUCCEED(error);

  clSetKernelArg(matmul_k, 0, sizeof(cl_int), &n);
  clSetKernelArg(matmul_k, 1, sizeof(cl_int), &m);
  clSetKernelArg(matmul_k, 2, sizeof(cl_int), &k);
  clSetKernelArg(matmul_k, 3, sizeof(cl_mem), &out);
  clSetKernelArg(matmul_k, 4, sizeof(cl_mem), &xss);
  clSetKernelArg(matmul_k, 5, sizeof(cl_mem), &yss);

  size_t local_work_size[2] = { 16, 16 };
  size_t global_work_size[2] =
    { div_rounding_up(n, local_work_size[0]) * local_work_size[0],
      div_rounding_up(k, local_work_size[1]) * local_work_size[1] };

  clEnqueueNDRangeKernel(queue,
                         matmul_k,
                         2,
                         NULL,
                         global_work_size,
                         local_work_size,
                         0, NULL, NULL);

  OPENCL_SUCCEED(clFinish(queue));
}
