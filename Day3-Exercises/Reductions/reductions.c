#include "../../clutils.h"

void benchmark_sequential_reduction(int n, cl_int *input, cl_int *output) {
  int result = 0;

  int64_t bef = get_wall_time();
  for (int i = 0; i < n; i++) {
    result += input[i];
  }
  int64_t aft = get_wall_time();

  printf("Sequential reduction: %dμs\n", (int)(aft-bef));

  *output = result;
}

void benchmark_tree_reduction(cl_context ctx, cl_command_queue queue, cl_device_id device,
                              cl_int n, cl_int *input, cl_int *output) {
  cl_int error = CL_SUCCESS;

  cl_program program = opencl_build_program(ctx, device, "kernels/tree_reduction.cl",
                                            "-Delem_t=int");
  cl_kernel tree_reduction_k = clCreateKernel(program, "tree_reduction", &error);
  OPENCL_SUCCEED(error);

  cl_mem mem_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                n*sizeof(cl_int), input, &error);
  OPENCL_SUCCEED(error);

  cl_mem mem_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                n*sizeof(cl_int), NULL, &error);
  OPENCL_SUCCEED(error);

  OPENCL_SUCCEED(clFinish(queue));

  int64_t bef = get_wall_time();
  while (n > 1) {
    int m = div_rounding_up(n, 2);

    size_t local_work_size[1] = { 256 };
    size_t global_work_size[1] = { div_rounding_up(m, local_work_size[0]) * local_work_size[0] };

    clSetKernelArg(tree_reduction_k, 0, sizeof(cl_int), &n);
    clSetKernelArg(tree_reduction_k, 1, sizeof(cl_int), &m);
    clSetKernelArg(tree_reduction_k, 2, sizeof(cl_mem), &mem_a);
    clSetKernelArg(tree_reduction_k, 3, sizeof(cl_mem), &mem_b);

    clEnqueueNDRangeKernel(queue, tree_reduction_k, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    n = m;

    cl_mem mem_c = mem_a;
    mem_a = mem_b;
    mem_b = mem_c;
  }
  OPENCL_SUCCEED(clFinish(queue));
  int64_t aft = get_wall_time();

  printf("Parallel reduction: %dμs\n", (int)(aft-bef));

  clEnqueueReadBuffer(queue, mem_a,
                      CL_TRUE,
                      0, sizeof(cl_int),
                      output,
                      0, NULL, NULL);

  OPENCL_SUCCEED(clReleaseMemObject(mem_a));
  OPENCL_SUCCEED(clReleaseMemObject(mem_b));
}

int main(int argc, char** argv) {
  cl_int n = 1000000;

  if (argc > 1) {
    n = atoi(argv[1]);
  }

  printf("Reduction over %d elements\n", n);

  cl_context ctx;
  cl_command_queue queue;
  cl_device_id device;

  opencl_init_command_queue_default(&device, &ctx, &queue);

  cl_int *input = calloc(n, sizeof(cl_int));

  for (int i = 0; i < n; i++) {
    input[i] = i;
  }

  cl_int correct, output;
  benchmark_sequential_reduction(n, input, &correct);

  benchmark_tree_reduction(ctx, queue, device, n, input, &output);
  assert(correct == output);
}
