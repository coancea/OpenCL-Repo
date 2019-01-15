// Simple un-tiled naively implemented matrix multiplication.  It
// does, however, use a trick to be generic in the type of elements
// multiplied.

#include "../clutils.h"

const int runs = 10;

void benchmark_matmul(cl_device_id device, cl_context ctx, cl_command_queue queue,
                      cl_int n, cl_int m, cl_int k,
                      char* elem_t, size_t sizeof_elem_t) {
  cl_int error;

  cl_program program = opencl_build_program(ctx, device, "kernels/matmul-naive.cl",
                                            "-Delem_t=%s", elem_t);
  cl_kernel matmul_k = clCreateKernel(program, "matmul", &error);

  cl_mem out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n*k*sizeof_elem_t, NULL, &error);
  OPENCL_SUCCEED(error);

  cl_mem xss = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n*m*sizeof_elem_t, NULL, &error);
  OPENCL_SUCCEED(error);

  cl_mem yss = clCreateBuffer(ctx, CL_MEM_READ_ONLY, m*k*sizeof_elem_t, NULL, &error);
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

  int64_t bef = get_wall_time();
  for (int i = 0; i < runs; i++) {
    clEnqueueNDRangeKernel(queue,
                           matmul_k,
                           2,
                           NULL,
                           global_work_size,
                           local_work_size,
                           0, NULL, NULL);
  }
  OPENCL_SUCCEED(clFinish(queue));

  int64_t aft = get_wall_time();
  int64_t elapsed_us = aft-bef;

  printf("%d kernels of total runtime %dμs (average %dμs)\n",
         runs, (int)elapsed_us, (int)(elapsed_us/runs));
}

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

  opencl_init_command_queue_default(&device, &ctx, &queue);

  printf("Element type int:\t");
  benchmark_matmul(device, ctx, queue,
                   n, m, k,
                   "int", sizeof(int));
  printf("Element type float:\t");
  benchmark_matmul(device, ctx, queue,
                   n, m, k,
                   "float", sizeof(float));
  printf("Element type double:\t");
  benchmark_matmul(device, ctx, queue,
                   n, m, k,
                   "double", sizeof(double));

}
