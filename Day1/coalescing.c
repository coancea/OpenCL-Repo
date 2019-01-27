// Demonstrating the impact of memory access patterns.

#include "../clutils.h"

int main(int argc, char **argv) {
  cl_int n = 10000;

  if (argc > 1) {
    n = atoi(argv[1]);
  }

  printf("Summing rows and columns of %d x %d matrix\n", n, n);

  cl_context ctx;
  cl_command_queue queue;
  cl_device_id device;
  cl_int error = CL_SUCCESS;

  opencl_init_command_queue_default(&device, &ctx, &queue);

  cl_program program = opencl_build_program(ctx, device, "kernels/coalescing.cl", "");
  cl_kernel sum_rows_k = clCreateKernel(program, "sum_rows", &error);
  cl_kernel sum_cols_k = clCreateKernel(program, "sum_cols", &error);
  OPENCL_SUCCEED(error);

  // We assume that the matrix is laid out in row-major order.
  cl_int *matrix = calloc(n*n, sizeof(cl_int));

  memset(matrix, 1, n*n*sizeof(cl_int));

  int runs = 10;
  int64_t bef, aft;

  // First, measure the time on the CPU.
  cl_int *sums = calloc(n, sizeof(cl_int));

  bef = get_wall_time();
  for (int i = 0; i < runs; i++) {
    for (int row = 0; row < n; row++) {
      cl_int sum = 0;
      for (int col = 0; col < n; col++) {
        sum += matrix[row*n+col];
      }
      sums[row] = sum;
    }
  }
  aft = get_wall_time();
  printf("Summing rows on CPU:\t %dμs\n", (int)((aft-bef)/runs));

  bef = get_wall_time();
  for (int i = 0; i < runs; i++) {
    for (int col = 0; col < n; col++) {
      cl_int sum = 0;
      for (int row = 0; row < n; row++) {
        sum += matrix[row*n+col];
      }
      sums[col] = sum;
    }
  }
  aft = get_wall_time();
  printf("Summing columns on CPU:\t %dμs\n", (int)((aft-bef)/runs));

  // Now for GPU timing.

  cl_mem input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                n*n*sizeof(cl_int), matrix, &error);
  OPENCL_SUCCEED(error);

  cl_mem output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n*sizeof(cl_int), NULL, &error);
  OPENCL_SUCCEED(error);

  clSetKernelArg(sum_cols_k, 0, sizeof(cl_int), &n);
  clSetKernelArg(sum_cols_k, 1, sizeof(cl_mem), &output);
  clSetKernelArg(sum_cols_k, 2, sizeof(cl_mem), &input);

  clSetKernelArg(sum_rows_k, 0, sizeof(cl_int), &n);
  clSetKernelArg(sum_rows_k, 1, sizeof(cl_mem), &output);
  clSetKernelArg(sum_rows_k, 2, sizeof(cl_mem), &input);

  size_t local_work_size[1] = { 256 };
  size_t global_work_size[1] = { div_rounding_up(n, local_work_size[0]) * local_work_size[0] };

  bef = get_wall_time();
  for (int i = 0; i < runs; i++) {
    clEnqueueNDRangeKernel(queue, sum_rows_k, 1, NULL,
                           global_work_size, local_work_size,
                           0, NULL, NULL);

  }

  OPENCL_SUCCEED(clFinish(queue));
  aft = get_wall_time();
  printf("Summing rows on GPU:\t %dμs\n", (int)((aft-bef)/runs));

  // Validate.
  cl_int *gpu_sums = calloc(n, sizeof(cl_int));
  clEnqueueReadBuffer(queue, output, CL_TRUE,
                      0, n * sizeof(cl_int),
                      gpu_sums,
                      0, NULL, NULL);

  if (memcmp(sums, gpu_sums, n * sizeof(cl_int)) != 0) {
    printf("Invalid result\n");
  }

  bef = get_wall_time();
  for (int i = 0; i < runs; i++) {
    clEnqueueNDRangeKernel(queue, sum_cols_k, 1, NULL,
                           global_work_size, local_work_size,
                           0, NULL, NULL);

  }

  OPENCL_SUCCEED(clFinish(queue));
  aft = get_wall_time();

  printf("Summing columns on GPU:\t %dμs\n", (int)((aft-bef)/runs));

  // Validate.
  gpu_sums = calloc(n, sizeof(cl_int));
  clEnqueueReadBuffer(queue, output, CL_TRUE,
                      0, n * sizeof(cl_int),
                      gpu_sums,
                      0, NULL, NULL);

  if (memcmp(sums, gpu_sums, n * sizeof(cl_int)) != 0) {
    printf("Invalid result\n");
  }
}
