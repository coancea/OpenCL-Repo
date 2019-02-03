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

int intmax(int y, int x) {
  if (y < x) {
    return x;
  } else {
    return y;
  }
}

cl_int4 mssp_mapf(int x) {
  cl_int4 v;
  int x0 = intmax(x, 0);
  v.s[0] = x0;
  v.s[1] = x0;
  v.s[2] = x0;
  v.s[3] = 0;
  return v;
}

cl_int4 mssp_redf(cl_int4 x, cl_int4 y) {
  cl_int4 r;
  r.s[0] = intmax(intmax(x.s[0], y.s[0]), x.s[2] + y.s[1]);
  r.s[1] = intmax(x.s[1], x.s[3] + y.s[1]);
  r.s[2] = intmax(x.s[2] + y.s[3], y.s[2]);
  r.s[3] = x.s[3] + y.s[3];
  return r;
}

void benchmark_sequential_mssp(int n, cl_int *input, cl_int *output) {
  cl_int4 result = mssp_mapf(0);

  int64_t bef = get_wall_time();
  for (int i = 0; i < n; i++) {
    result = mssp_redf(result, mssp_mapf(input[i]));
  }
  int64_t aft = get_wall_time();

  printf("Sequential MSSP:\t%dμs\n", (int)(aft-bef));

  *output = result.s[0];
}

void benchmark_tree_mssp(cl_context ctx, cl_command_queue queue, cl_device_id device,
                         cl_mem orig_mem_a, cl_mem orig_mem_b,
                         cl_int orig_n, cl_int *input, cl_int *output) {
  cl_int error = CL_SUCCESS;

  cl_program program = opencl_build_program(ctx, device, "kernels/tree_mssp.cl", "");
  cl_kernel tree_mssp_k = clCreateKernel(program, "tree_mssp", &error);
  OPENCL_SUCCEED(error);
  cl_kernel mssp_init_k = clCreateKernel(program, "mssp_init", &error);
  OPENCL_SUCCEED(error);

  cl_event *events = calloc(runs * (intlog(orig_n, 2)+1), sizeof(cl_event));
  int events_created = 0;

  cl_mem mem_a, mem_b, mem_c;

  for (int i = 0; i < runs; i++) {
    mem_a = orig_mem_a;
    mem_b = orig_mem_b;
    int n = orig_n;

    OPENCL_SUCCEED(clEnqueueWriteBuffer(queue, mem_a, CL_TRUE,
                                        0, n * sizeof(cl_int),
                                        input,
                                        0, NULL, NULL));

    // First we have to construct the quadruples from the original input.
    size_t init_local_work_size[1] = { 256 };
    size_t init_global_work_size[1] = { div_rounding_up(n, init_local_work_size[0]) * init_local_work_size[0] };
    clSetKernelArg(mssp_init_k, 0, sizeof(cl_int), &n);
    clSetKernelArg(mssp_init_k, 1, sizeof(cl_mem), &mem_a);
    clSetKernelArg(mssp_init_k, 2, sizeof(cl_mem), &mem_b);
    OPENCL_SUCCEED(clEnqueueNDRangeKernel(queue, mssp_init_k, 1,
                                          NULL, init_global_work_size, init_local_work_size,
                                          0, NULL, &events[events_created++]));

    // Swap mem_a and mem_b.
    mem_c = mem_a;
    mem_a = mem_b;
    mem_b = mem_c;

    while (n > 1) {
      int m = div_rounding_up(n, 2);

      size_t local_work_size[1] = { 256 };
      size_t global_work_size[1] = { div_rounding_up(m, local_work_size[0]) * local_work_size[0] };

      clSetKernelArg(tree_mssp_k, 0, sizeof(cl_int), &n);
      clSetKernelArg(tree_mssp_k, 1, sizeof(cl_int), &m);
      clSetKernelArg(tree_mssp_k, 2, sizeof(cl_mem), &mem_a);
      clSetKernelArg(tree_mssp_k, 3, sizeof(cl_mem), &mem_b);

      OPENCL_SUCCEED(clEnqueueNDRangeKernel(queue, tree_mssp_k, 1,
                                            NULL, global_work_size, local_work_size,
                                            0, NULL, &events[events_created++]));

      n = m;

      // Swap mem_a and mem_b.
      mem_c = mem_a;
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

  printf("Tree MSSP:      \t%dμs\n", time/runs);

  cl_int4 res;
  clEnqueueReadBuffer(queue, mem_a,
                      CL_TRUE,
                      0, sizeof(cl_int4),
                      &res,
                      0, NULL, NULL);
  *output = res.s[0];
}

void benchmark_group_mssp(cl_context ctx, cl_command_queue queue, cl_device_id device,
                          cl_mem orig_mem_a, cl_mem orig_mem_b,
                          cl_int orig_n, cl_int *input, cl_int *output) {
  cl_int error = CL_SUCCESS;

  cl_program program = opencl_build_program(ctx, device, "kernels/group_mssp.cl", "");
  cl_kernel group_mssp_k = clCreateKernel(program, "group_mssp", &error);
  OPENCL_SUCCEED(error);
  cl_kernel mssp_init_k = clCreateKernel(program, "mssp_init", &error);
  OPENCL_SUCCEED(error);

  size_t group_size = 256;
  cl_event *events = calloc(runs * intlog(orig_n, group_size), sizeof(cl_event));
  int events_created = 0;

  cl_mem mem_a, mem_b, mem_c;

  for (int i = 0; i < runs; i++) {
    mem_a = orig_mem_a;
    mem_b = orig_mem_b;
    int n = orig_n;

    OPENCL_SUCCEED(clEnqueueWriteBuffer(queue, mem_a, CL_TRUE,
                                        0, n * sizeof(cl_int),
                                        input,
                                        0, NULL, NULL));

    // First we have to construct the quadruples from the original input.
    size_t init_local_work_size[1] = { 256 };
    size_t init_global_work_size[1] = { div_rounding_up(n, init_local_work_size[0]) * init_local_work_size[0] };
    clSetKernelArg(mssp_init_k, 0, sizeof(cl_int), &n);
    clSetKernelArg(mssp_init_k, 1, sizeof(cl_mem), &mem_a);
    clSetKernelArg(mssp_init_k, 2, sizeof(cl_mem), &mem_b);
    OPENCL_SUCCEED(clEnqueueNDRangeKernel(queue, mssp_init_k, 1,
                                          NULL, init_global_work_size, init_local_work_size,
                                          0, NULL, &events[events_created++]));

    // Swap mem_a and mem_b.
    mem_c = mem_a;
    mem_a = mem_b;
    mem_b = mem_c;

    while (n > 1) {
      int m = div_rounding_up(n, group_size);

      size_t local_work_size[1] = { group_size };
      size_t global_work_size[1] = { m * local_work_size[0] };

      clSetKernelArg(group_mssp_k, 0, sizeof(cl_int), &n);
      clSetKernelArg(group_mssp_k, 1, sizeof(cl_mem), &mem_a);
      clSetKernelArg(group_mssp_k, 2, sizeof(cl_mem), &mem_b);
      clSetKernelArg(group_mssp_k, 3, local_work_size[0]*sizeof(cl_int4), NULL);

      OPENCL_SUCCEED(clEnqueueNDRangeKernel(queue, group_mssp_k, 1,
                                            NULL, global_work_size, local_work_size,
                                            0, NULL, &events[events_created++]));

      n = m;

      mem_c = mem_a;
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

  printf("Group MSSP:     \t%dμs\n", time/runs);

  cl_int4 res;
  clEnqueueReadBuffer(queue, mem_a,
                      CL_TRUE,
                      0, sizeof(cl_int4),
                      &res,
                      0, NULL, NULL);
  *output = res.s[0];
}

void benchmark_chunked_mssp(cl_context ctx, cl_command_queue queue, cl_device_id device,
                            cl_mem mem_a, cl_mem mem_b,
                            cl_int n, cl_int *input, cl_int *output) {
  cl_int error = CL_SUCCESS;

  cl_program program = opencl_build_program(ctx, device, "kernels/chunked_mssp.cl", "");
  cl_kernel chunked_mssp_stage_one_k = clCreateKernel(program, "chunked_mssp_stage_one", &error);
  OPENCL_SUCCEED(error);
  cl_kernel chunked_mssp_stage_two_k = clCreateKernel(program, "chunked_mssp_stage_two", &error);
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

    clSetKernelArg(chunked_mssp_stage_one_k, 0, sizeof(cl_int), &n);
    clSetKernelArg(chunked_mssp_stage_one_k, 1, sizeof(cl_mem), &mem_a);
    clSetKernelArg(chunked_mssp_stage_one_k, 2, sizeof(cl_mem), &mem_b);
    clSetKernelArg(chunked_mssp_stage_one_k, 3, stage_one_local_work_size[0]*sizeof(cl_int4), NULL);

    OPENCL_SUCCEED(clEnqueueNDRangeKernel(queue, chunked_mssp_stage_one_k, 1, NULL,
                                          stage_one_global_work_size,
                                          stage_one_local_work_size,
                                          0, NULL, &stage_one_events[i]));

    // Run a single-group kernel with mem_a and mem_b flipped.

    size_t stage_two_local_work_size[1] = { 64 };
    size_t stage_two_global_work_size[1] = { stage_two_local_work_size[0] };
    clSetKernelArg(chunked_mssp_stage_two_k, 0, sizeof(cl_int), &num_groups);
    clSetKernelArg(chunked_mssp_stage_two_k, 1, sizeof(cl_mem), &mem_b);
    clSetKernelArg(chunked_mssp_stage_two_k, 2, sizeof(cl_mem), &mem_a);
    clSetKernelArg(chunked_mssp_stage_two_k, 3, stage_two_local_work_size[0]*sizeof(cl_int4), NULL);

    OPENCL_SUCCEED(clEnqueueNDRangeKernel(queue, chunked_mssp_stage_two_k, 1, NULL,
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

  printf("Chunked MSSP:     \t%dμs\n", time/runs);


  cl_int4 res;
  clEnqueueReadBuffer(queue, mem_a,
                      CL_TRUE,
                      0, sizeof(cl_int4),
                      &res,
                      0, NULL, NULL);
  *output = res.s[0];
}

int main(int argc, char** argv) {
  cl_int n = 1000000;

  if (argc > 1) {
    n = atoi(argv[1]);
  }

  printf("Mssp over %d elements\n", n);

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
  benchmark_sequential_mssp(n, input, &correct);

  // Create memory here.  For easier correct functioning of the
  // chunked MSSP kernel, we make these buffers at minimum 1MiB
  // in size.  Further, we make room for cl_int4s.
  int k = 1024*1024 + n;

  cl_mem mem_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                k*sizeof(cl_int4), NULL, &error);
  OPENCL_SUCCEED(error);

  cl_mem mem_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                k*sizeof(cl_int4), NULL, &error);
  OPENCL_SUCCEED(error);

  OPENCL_SUCCEED(clFinish(queue));

  cl_int zero = 0;
  OPENCL_SUCCEED(clEnqueueFillBuffer(queue, mem_b, &zero, sizeof(cl_int), 0, k*sizeof(cl_int4),
                                     0, NULL, NULL));
  OPENCL_SUCCEED(clFinish(queue));
  benchmark_tree_mssp(ctx, queue, device,
                      mem_a, mem_b, n, input, &output);
  if (correct != output) {
    printf("Invalid result: got %d, expected %d\n", output, correct);
  }


  OPENCL_SUCCEED(clEnqueueFillBuffer(queue, mem_b, &zero, sizeof(cl_int), 0, k*sizeof(cl_int4),
                                     0, NULL, NULL));
  OPENCL_SUCCEED(clFinish(queue));
  benchmark_group_mssp(ctx, queue, device,
                       mem_a, mem_b,
                       n, input, &output);
  if (correct != output) {
    printf("Invalid result: got %d, expected %d\n", output, correct);
  }

  OPENCL_SUCCEED(clEnqueueFillBuffer(queue, mem_b, &zero, sizeof(cl_int), 0, k*sizeof(cl_int4),
                                     0, NULL, NULL));
  OPENCL_SUCCEED(clFinish(queue));
  benchmark_chunked_mssp(ctx, queue, device,
                         mem_a, mem_b,
                         n, input, &output);
  if (correct != output) {
    printf("Invalid result: got %d, expected %d\n", output, correct);
  }

  OPENCL_SUCCEED(clReleaseMemObject(mem_a));
  OPENCL_SUCCEED(clReleaseMemObject(mem_b));
}
