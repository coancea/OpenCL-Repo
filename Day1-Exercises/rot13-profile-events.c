// An example of how to profile an OpenCL program via the OpenCL event
// mechanism.

#include "../clutils.h"

int main(int argc, char** argv) {
  cl_int n = 100000;
  if (argc > 1) {
    n = atoi(argv[1]);
  }
  printf("Rot-13 on %d characters\n", n);

  cl_context ctx;
  cl_command_queue queue;
  cl_device_id device;
  cl_int error = CL_SUCCESS;

  opencl_init_command_queue_default(&device, &ctx, &queue);

  // Construct and build an OpenCL program from disk.
  cl_program program = opencl_build_program(ctx, device, "kernels/rot13.cl", "");

  // Construct the kernel from the program.
  cl_kernel rot13_k = clCreateKernel(program, "rot13", &error);
  OPENCL_SUCCEED(error);

  // Now we are ready to run.

  char *string = malloc(n+1);
  string[n] = 0;
  for (int i = 0; i < n; i++) {
    string[i] = i;
  }

  // Note: CL_MEM_READ_ONLY is only a restriction on kernels, not the host.
  cl_mem input = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n, NULL, &error);
  OPENCL_SUCCEED(error);

  cl_mem output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n, NULL, &error);
  OPENCL_SUCCEED(error);

  cl_event write_e;
  // Write the input to the kernel, asynchronously, constructing an
  // event that we can query for profiling information.
  OPENCL_SUCCEED(clEnqueueWriteBuffer(queue, input,
                                      CL_FALSE, // Non-blocking write
                                      0, // Offset in 'input'.
                                      n, // Number of bytes to copy.
                                      string, // Where to copy from.
                                      0, NULL, &write_e));

  clSetKernelArg(rot13_k, 0, sizeof(cl_mem), &output);
  clSetKernelArg(rot13_k, 1, sizeof(cl_mem), &input);
  clSetKernelArg(rot13_k, 2, sizeof(cl_int), &n);

  // Enqueue the rot13 kernel.
  size_t local_work_size[1] = { 256 };
  size_t global_work_size[1] = { div_rounding_up(n, local_work_size[0]) * local_work_size[0] };

  int runs = 10;

  // TODO: run and profile
}
