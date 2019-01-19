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

  printf("Game of Life on a %d by %d grid.\n", n, m);

  cl_context ctx;
  cl_command_queue queue;
  cl_device_id device;
  cl_int error = CL_SUCCESS;

  opencl_init_command_queue_default(&device, &ctx, &queue);

  cl_program program = opencl_build_program(ctx, device, "kernels/life-images.cl", "");

  cl_kernel life_k = clCreateKernel(program, "life", &error);
  OPENCL_SUCCEED(error);

  // Now we are ready to run.

  cl_int *cells = malloc(n * m * sizeof(cl_int));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      cells[i*m+j] = random() % 2;
    }
  }

  // We want a simple format with four 8-bit channels per element (but
  // it is not important, because we are not really encoding colours
  // for this program).  The only important thing is that it matches sizeof(cl_int).
  cl_image_format format =
    { .image_channel_order = CL_RGBA,
      .image_channel_data_type = CL_UNSIGNED_INT8
    };

  cl_image_desc desc =
    { .image_type =  CL_MEM_OBJECT_IMAGE2D, // 2D image.
      .image_width = m,
      .image_height = n,
      .image_depth = 1, // Not used in 2D case.
      .image_array_size = 0, // Only used for image arrays.
      .image_row_pitch = m * sizeof(cl_int), // Pitch per row.
      .image_slice_pitch = 0, // Only used for 3D images and image arrays.
      .num_mip_levels = 0, // Must be 0.
      .num_samples = 0, // Must be 0.
      .buffer = NULL // Only used for CL_MEM_OBJECT_IMAGE1D_BUFFER.
    };

  cl_mem mem_a = clCreateImage(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               &format, &desc,
                               cells, &error);
  OPENCL_SUCCEED(error);

  cl_mem mem_b = clCreateImage(ctx, CL_MEM_READ_WRITE,
                               &format, &desc,
                               NULL, &error);
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
    clEnqueueNDRangeKernel(queue,
                           life_k, // The kernel.
                           2, // Number of grid dimensions.
                           NULL, // Must always be NULL (supposed to
                                 // be for grid offset).
                           global_work_size,
                           local_work_size,
                           0, NULL, NULL);
    cl_mem mem_c = mem_a;
    mem_a = mem_b;
    mem_b = mem_c;
  }

  // Wait for the kernel to stop.
  OPENCL_SUCCEED(clFinish(queue));

  int64_t aft = get_wall_time();
  int64_t elapsed_us = aft-bef;

  // Describe what we want to copy from the image.
  const size_t origin[] = { 0, 0, 0 };
  const size_t region[] = { n, m, 1 };

  clEnqueueReadImage(queue, mem_a,
                     CL_TRUE,
                     origin, region,
                     desc.image_row_pitch,
                     desc.image_slice_pitch,
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
