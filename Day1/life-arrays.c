// Game of life with arrays.

#include "../clutils.h"

int main(int argc, char** argv) {
  cl_int n = 1000, m = 1000, iters = 10;

  if (argc > 1) {
    n = atoi(argv[1]);
  }

  if (argc > 2) {
    m = atoi(argv[2]);
  }

  if (argc > 3) {
    iters = atoi(argv[3]);
  }

  printf("Game of Life on a %d by %d grid; %d iterations.\n", n, m, iters);

  cl_context ctx;
  cl_command_queue queue;
  cl_device_id device;
  cl_int error = CL_SUCCESS;

  opencl_init_command_queue_default(&device, &ctx, &queue);

  cl_program program = opencl_build_program(ctx, device, "kernels/life-arrays.cl", "");

  cl_kernel life_k = clCreateKernel(program, "life", &error);
  OPENCL_SUCCEED(error);

  // Now we are ready to run.

  cl_int *cells = malloc(n * m * sizeof(cl_int));

  srand(123); // Ensure predictable image.
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cells[i*n+j] = rand() % 2;
    }
  }

  cl_mem mem_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                n * m * sizeof(cl_int), cells, &error);
  OPENCL_SUCCEED(error);

  cl_mem mem_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                n * m * sizeof(cl_int), NULL, &error);
  OPENCL_SUCCEED(error);

  clSetKernelArg(life_k, 0, sizeof(cl_int), &n);
  clSetKernelArg(life_k, 1, sizeof(cl_int), &m);

  // Enqueue the rot13 kernel.
  size_t local_work_size[2] = { 16, 16 };
  size_t global_work_size[2] = { div_rounding_up(n, local_work_size[1]) * local_work_size[1],
                                 div_rounding_up(m, local_work_size[1]) * local_work_size[1]};

  int64_t bef = get_wall_time();
  for (int i = 0; i < iters; i++) {
    clSetKernelArg(life_k, 2, sizeof(cl_mem), &mem_a);
    clSetKernelArg(life_k, 3, sizeof(cl_mem), &mem_b);
    OPENCL_SUCCEED(clEnqueueNDRangeKernel(queue,
                                          life_k, // The kernel.
                                          2, // Number of grid dimensions.
                                          NULL, // Must always be NULL (supposed to
                                          // be for grid offset).
                                          global_work_size,
                                          local_work_size,
                                          0, NULL, NULL));
    cl_mem mem_c = mem_a;
    mem_a = mem_b;
    mem_b = mem_c;
  }

  // Wait for the kernel to stop.
  OPENCL_SUCCEED(clFinish(queue));

  int64_t aft = get_wall_time();
  int64_t elapsed_us = aft-bef;

  clEnqueueReadBuffer(queue, mem_a,
                      CL_TRUE,
                      0, n*m*sizeof(cl_int),
                      cells,
                      0, NULL, NULL);

  if (n < 100 && m < 100) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        printf("%c", cells[i*n+j] ? '#' : ' ');
      }
      printf("\n");
    }
  }

  if (iters > 0) {
    printf("%d kernels of total runtime %dμs (average %dμs)\n",
           iters, (int)elapsed_us, (int)(elapsed_us/iters));
  }
}
