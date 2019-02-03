#include "../../clutils.h"

const int runs = 100;

int intlog(int x, int base) {
  int y = 0;
  while (x > 0) {
    y++;
    x /= base;
  }
  return y;
}

int event_runtime_us(cl_event e) {
  cl_ulong t_start, t_end;
  OPENCL_SUCCEED(clGetEventProfilingInfo(e,
                                         CL_PROFILING_COMMAND_START,
                                         sizeof(t_start), &t_start,
                                         NULL));
  OPENCL_SUCCEED(clGetEventProfilingInfo(e,
                                         CL_PROFILING_COMMAND_END,
                                         sizeof(t_end), &t_end,
                                         NULL));

  return (t_end - t_start) / 1000;
}

void benchmark_sequential_reduction(int n, cl_int *input, cl_int *output) {
  int result = 0;

  int64_t bef = get_wall_time();
  for (int i = 0; i < n; i++) {
    result += input[i];
  }
  int64_t aft = get_wall_time();

  printf("Sequential reduction:\t%dμs\n", (int)(aft-bef));

  *output = result;
}

void benchmark_tree_reduction(cl_context ctx, cl_command_queue queue, cl_device_id device,
                              cl_mem orig_mem_a, cl_mem orig_mem_b,
                              cl_int orig_n, cl_int *input, cl_int *output) {
  cl_int error = CL_SUCCESS;

  cl_program program = opencl_build_program(ctx, device, "kernels/tree_reduction.cl",
                                            "-Delem_t=int");
  cl_kernel tree_reduction_k = clCreateKernel(program, "tree_reduction", &error);
  OPENCL_SUCCEED(error);


  cl_event *events = calloc(runs * intlog(orig_n, 2), sizeof(cl_event));
  int events_created = 0;

  cl_mem mem_a, mem_b;

  for (int i = 0; i < runs; i++) {
    mem_a = orig_mem_a;
    mem_b = orig_mem_b;
    int n = orig_n;

    OPENCL_SUCCEED(clEnqueueWriteBuffer(queue, mem_a, CL_TRUE,
                                      0, n * sizeof(cl_int),
                                      input,
                                      0, NULL, NULL));

    while (n > 1) {
      int m = div_rounding_up(n, 2);

      size_t local_work_size[1] = { 256 };
      size_t global_work_size[1] = { div_rounding_up(m, local_work_size[0]) * local_work_size[0] };

      clSetKernelArg(tree_reduction_k, 0, sizeof(cl_int), &n);
      clSetKernelArg(tree_reduction_k, 1, sizeof(cl_int), &m);
      clSetKernelArg(tree_reduction_k, 2, sizeof(cl_mem), &mem_a);
      clSetKernelArg(tree_reduction_k, 3, sizeof(cl_mem), &mem_b);

      clEnqueueNDRangeKernel(queue, tree_reduction_k, 1, NULL, global_work_size, local_work_size, 0, NULL, &events[events_created++]);

      n = m;

      cl_mem mem_c = mem_a;
      mem_a = mem_b;
      mem_b = mem_c;
    }
  }
  OPENCL_SUCCEED(clFinish(queue));


  int time = 0;
  for (int i = 0; i < events_created; i++) {
    time += event_runtime_us(events[i]);
    OPENCL_SUCCEED(clReleaseEvent(events[i]));
  }
  free(events);

  printf("Tree reduction:  \t%dμs\n", time/runs);

  clEnqueueReadBuffer(queue, mem_a,
                      CL_TRUE,
                      0, sizeof(cl_int),
                      output,
                      0, NULL, NULL);
}

void benchmark_group_reduction(cl_context ctx, cl_command_queue queue, cl_device_id device,
                               cl_mem orig_mem_a, cl_mem orig_mem_b,
                               cl_int orig_n, cl_int *input, cl_int *output) {
  cl_int error = CL_SUCCESS;

  cl_program program = opencl_build_program(ctx, device, "kernels/group_reduction.cl",
                                            "-Delem_t=int");
  cl_kernel group_reduction_k = clCreateKernel(program, "group_reduction", &error);
  OPENCL_SUCCEED(error);

  size_t group_size = 256;
  cl_event *events = calloc(runs * intlog(orig_n, group_size), sizeof(cl_event));
  int events_created = 0;

  cl_mem mem_a, mem_b;

  for (int i = 0; i < runs; i++) {
    mem_a = orig_mem_a;
    mem_b = orig_mem_b;
    int n = orig_n;

    OPENCL_SUCCEED(clEnqueueWriteBuffer(queue, mem_a, CL_TRUE,
                                        0, n * sizeof(cl_int),
                                        input,
                                        0, NULL, NULL));

    while (n > 1) {
      int m = div_rounding_up(n, group_size);

      size_t local_work_size[1] = { group_size };
      size_t global_work_size[1] = { m * local_work_size[0] };

      clSetKernelArg(group_reduction_k, 0, sizeof(cl_int), &n);
      clSetKernelArg(group_reduction_k, 1, sizeof(cl_mem), &mem_a);
      clSetKernelArg(group_reduction_k, 2, sizeof(cl_mem), &mem_b);
      clSetKernelArg(group_reduction_k, 3, local_work_size[0]*sizeof(cl_int), NULL);

      OPENCL_SUCCEED(clEnqueueNDRangeKernel(queue, group_reduction_k, 1, NULL, global_work_size, local_work_size, 0, NULL, &events[events_created++]));

      n = m;

      cl_mem mem_c = mem_a;
      mem_a = mem_b;
      mem_b = mem_c;
    }
  }
  OPENCL_SUCCEED(clFinish(queue));

  int time = 0;
  for (int i = 0; i < events_created; i++) {
    time += event_runtime_us(events[i]);
    OPENCL_SUCCEED(clReleaseEvent(events[i]));
  }
  free(events);

  printf("Group reduction:\t%dμs\n", time/runs);

  clEnqueueReadBuffer(queue, mem_a,
                      CL_TRUE,
                      0, sizeof(cl_int),
                      output,
                      0, NULL, NULL);
}

void benchmark_chunked_reduction(cl_context ctx, cl_command_queue queue, cl_device_id device,
                                 cl_mem mem_a, cl_mem mem_b,
                                 cl_int n, cl_int *input, cl_int *output) {
  cl_int error = CL_SUCCESS;

  cl_program program = opencl_build_program(ctx, device, "kernels/chunked_reduction.cl",
                                            "-Delem_t=int");
  cl_kernel chunked_reduction_k = clCreateKernel(program, "chunked_reduction", &error);
  OPENCL_SUCCEED(error);

  cl_event *stage_one_events = calloc(runs, sizeof(cl_event));
  cl_event *stage_two_events = calloc(runs, sizeof(cl_event));

  for (int i = 0; i < runs; i++) {
    OPENCL_SUCCEED(clEnqueueWriteBuffer(queue, mem_a, CL_TRUE,
                                        0, n * sizeof(cl_int),
                                        input,
                                        0, NULL, NULL));

    cl_int num_groups = 1024;
    size_t stage_one_local_work_size[1] = { 256 };
    size_t stage_one_global_work_size[1] = { num_groups * stage_one_local_work_size[0] };

    clSetKernelArg(chunked_reduction_k, 0, sizeof(cl_int), &n);
    clSetKernelArg(chunked_reduction_k, 1, sizeof(cl_mem), &mem_a);
    clSetKernelArg(chunked_reduction_k, 2, sizeof(cl_mem), &mem_b);
    clSetKernelArg(chunked_reduction_k, 3, stage_one_local_work_size[0]*sizeof(cl_int), NULL);

    OPENCL_SUCCEED(clEnqueueNDRangeKernel(queue, chunked_reduction_k, 1, NULL,
                                          stage_one_global_work_size,
                                          stage_one_local_work_size,
                                          0, NULL, &stage_one_events[i]));

    // Run a single-group kernel with mem_a and mem_b flipped.

    size_t stage_two_local_work_size[1] = { 64 };
    size_t stage_two_global_work_size[1] = { stage_two_local_work_size[0] };
    clSetKernelArg(chunked_reduction_k, 0, sizeof(cl_int), &num_groups);
    clSetKernelArg(chunked_reduction_k, 1, sizeof(cl_mem), &mem_b);
    clSetKernelArg(chunked_reduction_k, 2, sizeof(cl_mem), &mem_a);
    clSetKernelArg(chunked_reduction_k, 3, stage_two_local_work_size[0]*sizeof(cl_int), NULL);

    OPENCL_SUCCEED(clEnqueueNDRangeKernel(queue, chunked_reduction_k, 1, NULL,
                                          stage_two_global_work_size,
                                          stage_two_local_work_size,
                                          0, NULL, &stage_two_events[i]));
  }
  OPENCL_SUCCEED(clFinish(queue));

  int time = 0;
  for (int i = 0; i < runs; i++) {
    time += event_runtime_us(stage_one_events[i]);
    OPENCL_SUCCEED(clReleaseEvent(stage_one_events[i]));
    time += event_runtime_us(stage_two_events[i]);
    OPENCL_SUCCEED(clReleaseEvent(stage_two_events[i]));
  }
  free(stage_one_events);
  free(stage_two_events);

  printf("Chunked reduction:\t%dμs\n", time/runs);

  clEnqueueReadBuffer(queue, mem_a,
                      CL_TRUE,
                      0, sizeof(cl_int),
                      output,
                      0, NULL, NULL);
}

void benchmark_atomic_reduction(cl_context ctx, cl_command_queue queue, cl_device_id device,
                                cl_mem mem_a, cl_mem mem_b,
                                cl_int n, cl_int *input, cl_int *output) {
  cl_int error = CL_SUCCESS;

  cl_program program = opencl_build_program(ctx, device, "kernels/atomic_reduction.cl",
                                            "");
  cl_kernel atomic_reduction_k = clCreateKernel(program, "atomic_reduction", &error);
  OPENCL_SUCCEED(error);

  OPENCL_SUCCEED(clEnqueueWriteBuffer(queue, mem_a, CL_TRUE,
                                      0, n * sizeof(cl_int),
                                      input,
                                      0, NULL, NULL));

  size_t local_work_size[1] = { 256 };
  size_t global_work_size[1] = { div_rounding_up(n, local_work_size[0]) * local_work_size[0] };

  clSetKernelArg(atomic_reduction_k, 0, sizeof(cl_int), &n);
  clSetKernelArg(atomic_reduction_k, 1, sizeof(cl_mem), &mem_a);
  clSetKernelArg(atomic_reduction_k, 2, sizeof(cl_mem), &mem_b);

  cl_event *events = calloc(runs, sizeof(cl_event));

  for (int i = 0; i < runs; i++) {
    // First we zero the accumulator.
    cl_int zero = 0;
    OPENCL_SUCCEED(clEnqueueWriteBuffer(queue, mem_b, CL_TRUE,
                                        0, sizeof(cl_int),
                                        &zero,
                                        0, NULL, NULL));

    OPENCL_SUCCEED(clEnqueueNDRangeKernel(queue, atomic_reduction_k, 1, NULL,
                                          global_work_size,
                                          local_work_size,
                                          0, NULL, &events[i]));
  }
  OPENCL_SUCCEED(clFinish(queue));

  int time = 0;
  for (int i = 0; i < runs; i++) {
    time += event_runtime_us(events[i]);
    OPENCL_SUCCEED(clReleaseEvent(events[i]));
  }
  free(events);

  printf("Atomic reduction:\t%dμs\n", time/runs);

  clEnqueueReadBuffer(queue, mem_b,
                      CL_TRUE,
                      0, sizeof(cl_int),
                      output,
                      0, NULL, NULL);
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

  cl_int error;
  cl_int correct, output;
  benchmark_sequential_reduction(n, input, &correct);

  // Create memory here.  For easier correct functioning of the
  // chunked reduction kernel, we make these buffers at minimum 1MiB
  // in size.
  int k = 1024*1024 + n;

  cl_mem mem_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                k*sizeof(cl_int), NULL, &error);
  OPENCL_SUCCEED(error);

  cl_mem mem_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                k*sizeof(cl_int), NULL, &error);
  OPENCL_SUCCEED(error);

  OPENCL_SUCCEED(clFinish(queue));

  cl_int zero = 0;
  OPENCL_SUCCEED(clEnqueueFillBuffer(queue, mem_b, &zero, sizeof(cl_int), 0, k*sizeof(cl_int),
                                     0, NULL, NULL));
  OPENCL_SUCCEED(clFinish(queue));
  benchmark_tree_reduction(ctx, queue, device,
                           mem_a, mem_b, n, input, &output);
  if (correct != output) {
    printf("Invalid result: got %d, expected %d\n", output, correct);
  }


  OPENCL_SUCCEED(clEnqueueFillBuffer(queue, mem_b, &zero, sizeof(cl_int), 0, k*sizeof(cl_int),
                                     0, NULL, NULL));
  OPENCL_SUCCEED(clFinish(queue));
  benchmark_group_reduction(ctx, queue, device,
                            mem_a, mem_b,
                            n, input, &output);
  if (correct != output) {
    printf("Invalid result: got %d, expected %d\n", output, correct);
  }

  OPENCL_SUCCEED(clEnqueueFillBuffer(queue, mem_b, &zero, sizeof(cl_int), 0, k*sizeof(cl_int),
                                     0, NULL, NULL));
  OPENCL_SUCCEED(clFinish(queue));
  benchmark_chunked_reduction(ctx, queue, device,
                              mem_a, mem_b,
                              n, input, &output);
  if (correct != output) {
    printf("Invalid result: got %d, expected %d\n", output, correct);
  }

  OPENCL_SUCCEED(clEnqueueFillBuffer(queue, mem_b, &zero, sizeof(cl_int), 0, k*sizeof(cl_int),
                                     0, NULL, NULL));
  OPENCL_SUCCEED(clFinish(queue));
  benchmark_atomic_reduction(ctx, queue, device,
                             mem_a, mem_b,
                             n, input, &output);
  if (correct != output) {
    printf("Invalid result: got %d, expected %d\n", output, correct);
  }

  OPENCL_SUCCEED(clReleaseMemObject(mem_a));
  OPENCL_SUCCEED(clReleaseMemObject(mem_b));
}
