// A demonstration of the effects of branch divergence.

#include "../clutils.h"

void do_fibfact_on_device(cl_context ctx, cl_command_queue queue,
                          int k, cl_kernel fibfact_k,
                          cl_int *ns_host, cl_int *ops_host) {

  // TODO: move data to the GPU.

  int runs = 10;

  int64_t bef = get_wall_time();
  for (int i = 0; i < runs; i++) {
    // TODO: launch the kernel.
  }

  // Wait for the kernel(s) to stop.
  OPENCL_SUCCEED(clFinish(queue));

  int64_t aft = get_wall_time();
  int64_t elapsed_us = aft-bef;

  printf("%d kernels of total runtime %dμs (average %dμs)\n",
         runs, (int)elapsed_us, (int)(elapsed_us/runs));

  // TODO: Maybe clean up.
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
}
