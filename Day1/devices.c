// This program prints the indexes and names of all OpenCL platforms
// and their corresponding devices.

#include "../clutils.h"

int main() {
  cl_uint num_platforms;

  // Find the number of platforms.
  OPENCL_SUCCEED(clGetPlatformIDs(0, NULL, &num_platforms));

  printf("Found %d platforms\n", (int)num_platforms);

  // Make room for them.
  cl_platform_id *all_platforms = calloc(num_platforms, sizeof(cl_platform_id));

  // Fetch all the platforms.
  OPENCL_SUCCEED(clGetPlatformIDs(num_platforms, all_platforms, NULL));

  for (unsigned int i = 0; i < num_platforms; i++) {
    size_t req_bytes;
    char *name;

    // How much space do we need for the platform name?
    OPENCL_SUCCEED(clGetPlatformInfo(all_platforms[i], CL_PLATFORM_NAME, 0, NULL, &req_bytes));

    // Allocate space for the name and fetch it.
    name = malloc(req_bytes);
    OPENCL_SUCCEED(clGetPlatformInfo(all_platforms[i], CL_PLATFORM_NAME, req_bytes, name, NULL));

    printf("Platform %d: %s\n", i, name);

    free(name);

    // Now let us print the names of all the devices.  First we count
    // how many of them exist.
    cl_uint num_devices;
    OPENCL_SUCCEED(clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));

    // Then we make room for them.
    cl_device_id *platform_devices = calloc(num_devices, sizeof(cl_device_id));

    // Then we fetch them.
    OPENCL_SUCCEED(clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL,
                                  num_devices, platform_devices, NULL));

    for (unsigned int j = 0; j < num_devices; j++) {
      // How much space do we need for the device name?
      OPENCL_SUCCEED(clGetDeviceInfo(platform_devices[j], CL_DEVICE_NAME,
                                     0, NULL, &req_bytes));

      // Allocate space for the name and fetch it.
      name = malloc(req_bytes);
      OPENCL_SUCCEED(clGetDeviceInfo(platform_devices[j], CL_DEVICE_NAME,
                                     req_bytes, name, NULL));

      printf("\tDevice %d: %s\n", j, name);
      free(name);
    }
  }
}
