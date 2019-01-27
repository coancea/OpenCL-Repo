// A demonstration of the effects of branch divergence.

#include "../clutils.h"

void do_fibfact_on_device(cl_context ctx, cl_command_queue queue,
                          int k, cl_kernel fibfact_k,
                          cl_int *ns_host, cl_int *ops_host) {
  cl_int error = CL_SUCCESS;

  // Instead of using clEnqueueWriteBuffer(), we use
  // CL_MEM_COPY_HOST_PTR to initialise the memory objects from host
  // memory.

  cl_mem ns_dev = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 k*sizeof(cl_int), ns_host, &error);
  OPENCL_SUCCEED(error);
  cl_mem ops_dev = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  k*sizeof(cl_int), ops_host, &error);
  OPENCL_SUCCEED(error);
  cl_mem res_dev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, k*sizeof(cl_float), NULL, &error);
  OPENCL_SUCCEED(error);

  OPENCL_SUCCEED(clFinish(queue));

  clSetKernelArg(fibfact_k, 0, sizeof(cl_int), &k);
  clSetKernelArg(fibfact_k, 1, sizeof(cl_mem), &res_dev);
  clSetKernelArg(fibfact_k, 2, sizeof(cl_mem), &ns_dev);
  clSetKernelArg(fibfact_k, 3, sizeof(cl_mem), &ops_dev);

  size_t local_work_size[1] = { 256 };
  size_t global_work_size[1] = { div_rounding_up(k, local_work_size[0]) * local_work_size[0] };

  int runs = 10;

  int64_t bef = get_wall_time();
  for (int i = 0; i < runs; i++) {
    clEnqueueNDRangeKernel(queue, fibfact_k, 1, NULL,
                           global_work_size, local_work_size,
                           0, NULL, NULL);
  }

  // Wait for the kernel(s) to stop.
  OPENCL_SUCCEED(clFinish(queue));

  int64_t aft = get_wall_time();
  int64_t elapsed_us = aft-bef;

  printf("%d kernels of total runtime %dμs (average %dμs)\n",
         runs, (int)elapsed_us, (int)(elapsed_us/runs));

  OPENCL_SUCCEED(clReleaseMemObject(ns_dev));
  OPENCL_SUCCEED(clReleaseMemObject(ops_dev));
  OPENCL_SUCCEED(clReleaseMemObject(res_dev));
}

// For use in qsort().
int cmp_cl_int(const void *px, const void *py) {
  cl_int x = *(cl_int*)px;
  cl_int y = *(cl_int*)py;

  return x - y;
}

int main(int argc, char** argv) {
  cl_int k = 100000;
  if (argc > 1) {
    k = atoi(argv[1]);
  }

  cl_context ctx;
  cl_command_queue queue;
  cl_device_id device;
  cl_int error = CL_SUCCESS;

  opencl_init_command_queue_default(&device, &ctx, &queue);

  cl_program program = opencl_build_program(ctx, device, "kernels/fibfact.cl", "");
  cl_kernel fibfact_k = clCreateKernel(program, "fibfact", &error);

  int n_min = 1000, n_max = 5000;

  cl_int *ns_host = calloc(k, sizeof(int));
  cl_int *ops_host = calloc(k, sizeof(int));
  for (int i = 0; i < k; i++) {
    ns_host[i] = rand()%(n_max-n_min)+n_min;
    ops_host[i] = rand()%2;
  }

  printf("Random ns and random ops\n");
  do_fibfact_on_device(ctx, queue, k, fibfact_k, ns_host, ops_host);

  printf("Random ns and sorted ops\n");
  qsort(ops_host, k, sizeof(cl_int), cmp_cl_int);
  do_fibfact_on_device(ctx, queue, k, fibfact_k, ns_host, ops_host);

  printf("Sorted ns and sorted ops\n");
  qsort(ns_host, k, sizeof(cl_int), cmp_cl_int);
  do_fibfact_on_device(ctx, queue, k, fibfact_k, ns_host, ops_host);
}
