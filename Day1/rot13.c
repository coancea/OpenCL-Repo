// A very simple program showing how to start up OpenCL and run a
// simple kernel.

#include "../clutils.h"

int main() {
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

  char *string = "Hello, World!\n";
  cl_int n = strlen(string);

  // Note: CL_MEM_READ_ONLY is only a restriction on kernels, not the host.
  cl_mem input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                n, string, &error);
  OPENCL_SUCCEED(error);

  cl_mem output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n, NULL, &error);
  OPENCL_SUCCEED(error);

  clSetKernelArg(rot13_k, 0, sizeof(cl_mem), &output);
  clSetKernelArg(rot13_k, 1, sizeof(cl_mem), &input);
  clSetKernelArg(rot13_k, 2, sizeof(cl_int), &n);

  // Enqueue the rot13 kernel.
  size_t local_work_size[1] = { 256 };
  size_t global_work_size[1] = { div_rounding_up(n, local_work_size[0]) * local_work_size[0] };
  clEnqueueNDRangeKernel(queue,
                         rot13_k, // The kernel.
                         1, // Number of grid dimensions.
                         NULL, // Must always be NULL (supposed to be
                               // for grid offset).
                         global_work_size,
                         local_work_size,
                         0, NULL, NULL);

  // Wait for the kernel to stop.
  OPENCL_SUCCEED(clFinish(queue));

  // Read back the result.
  char *output_string = malloc(n + 1);
  output_string[n] = '\0'; // Ensure 0-termination.
  clEnqueueReadBuffer(queue, output,
                      CL_TRUE, // Blocking read.
                      0, n, // Offset zero in GPU memory, n bytes.
                      output_string, // Where to write on the host.
                      0, NULL, NULL);

  printf("Result: %s\n", output_string);
}
