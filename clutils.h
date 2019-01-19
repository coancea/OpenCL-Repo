// A header-only collection of lightweight helper functions for
// OpenCL.  This is intentionally not a "framework", and you will
// still need to understand and use the ordinary OpenCL C API.
//
// This header files takes care of including cl.h as appropriate for
// the operating system (although include paths and the like must
// still be setup correctly).  All functions defined in this file are
// prefixed with 'opencl_' and use snake_case, to distinguish from
// standard OpenCL functions that use a 'cl' prefix and camelCase.
//
// For pedagogical purposes, we have tried to comment this header file
// extensively.

// For compatibility with older OpenCLs, we will be using some
// deprecated operations.  This can be noisy unless we say that we
// really mean it!
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_SILENCE_DEPRECATION // For macOS.

// Apple keeps the cl.h in a nonstandard location for some reason.
#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

// Various other utility headers.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

//// Non-OpenCL utility functions.

// Read a file into a NUL-terminated string; returns NULL on error.
char* slurp_file(const char *filename) {
  char *s;
  FILE *f = fopen(filename, "rb"); // To avoid Windows messing with linebreaks.
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  size_t src_size = ftell(f);
  fseek(f, 0, SEEK_SET);
  s = (char*) malloc(src_size + 1);
  if (fread(s, 1, src_size, f) != src_size) {
    free(s);
    s = NULL;
  } else {
    s[src_size] = '\0';
  }
  fclose(f);

  return s;
}

// Unsigned integer division, rounding up.
size_t div_rounding_up(size_t x, size_t y) {
  return (x + y - 1) / y;
}

// Terminate the process with given error code, but first printing the
// given printf()-style error message.
static void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

//// Some rough timing support.  OpenCL itself supports more
// fine-grained timing and profiling, but often we are interested in
// wall clock measurements
//
// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

//// Now for OpenCL.

// Transform an OpenCL error code to an error message.
static const char* opencl_error_string(unsigned int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}

// If the error code is not CL_SUCCESS, terminate the process with an
// error message.  This function is not intended to be used directly,
// but instead via the OPENCL_SUCCEED macro.
static void opencl_succeed_fatal(unsigned int ret,
                                 const char *call,
                                 const char *file,
                                 int line) {
  if (ret != CL_SUCCESS) {
    panic(-1, "%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
          file, line, call, ret, opencl_error_string(ret));
  }
}

// Terminate the process with an error message unless the argument is CL_SUCCESS.
#define OPENCL_SUCCEED(e) opencl_succeed_fatal(e, #e, __FILE__, __LINE__)

static char* opencl_platform_name(cl_platform_id platform) {
  size_t req_bytes;
  char *name;

  OPENCL_SUCCEED(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &req_bytes));

  name = malloc(req_bytes);
  OPENCL_SUCCEED(clGetPlatformInfo(platform, CL_PLATFORM_NAME, req_bytes, name, NULL));

  return name;
}

static char* opencl_device_name(cl_device_id device) {
  size_t req_bytes;
  char *name;

  OPENCL_SUCCEED(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &req_bytes));

  name = malloc(req_bytes);
  OPENCL_SUCCEED(clGetDeviceInfo(device, CL_DEVICE_NAME, req_bytes, name, NULL));

  return name;
}

// Create a context and command queue for the given platform and
// device ID.  Terminates the program on error.
static void opencl_init_command_queue(unsigned int platform_index, unsigned int device_index,
                                      cl_device_id *device, cl_context *ctx, cl_command_queue *queue) {
  cl_uint num_platforms;
  OPENCL_SUCCEED(clGetPlatformIDs(0, NULL, &num_platforms));
  cl_platform_id *all_platforms = (cl_platform_id*) calloc(num_platforms, sizeof(cl_platform_id));
  OPENCL_SUCCEED(clGetPlatformIDs(num_platforms, all_platforms, NULL));

  assert(platform_index < num_platforms);
  cl_platform_id platform = all_platforms[platform_index];

  cl_uint num_devices;
  OPENCL_SUCCEED(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));
  cl_device_id *platform_devices = (cl_device_id*) calloc(num_devices, sizeof(cl_device_id));
  OPENCL_SUCCEED(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                num_devices, platform_devices, NULL));

  assert(device_index < num_devices);
  *device = platform_devices[device_index];

  char *platform_name = opencl_platform_name(platform);
  char *device_name = opencl_device_name(*device);
  printf("Using platform: %s\nUsing device: %s\n", platform_name, device_name);
  free(platform_name);
  free(device_name);

  // NVIDIA's OpenCL requires the platform property
  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)platform,
    0
  };

  cl_int error;
  *ctx = clCreateContext(properties, 1, device, NULL, NULL, &error);
  OPENCL_SUCCEED(error);

  *queue = clCreateCommandQueue(*ctx, *device, 0, &error);
  OPENCL_SUCCEED(error);

  free(all_platforms);
}

// As opencl_init_command_queue(), but picking platform and device by
// magic and/or configuration.
static void opencl_init_command_queue_default(cl_device_id *device, cl_context *ctx, cl_command_queue *queue) {
  const char* platform_id_str = getenv("OPENCL_PLATFORM_ID");
  const char* device_id_str = getenv("OPENCL_DEVICE_ID");

  int platform_id = platform_id_str ? atoi(platform_id_str) : 0;
  int device_id = device_id_str ? atoi(device_id_str) : 0;

  opencl_init_command_queue(platform_id, device_id, device, ctx, queue);
}

static cl_program opencl_build_program(cl_context ctx, cl_device_id device,
                                       const char *file, const char *options_fmt, ...) {
  cl_int error;
  const char *kernel_src = slurp_file(file);
  if (kernel_src == NULL) {
    fprintf(stderr, "Cannot open %s: %s\n", file, strerror(errno));
    abort();
  }
  size_t src_len = strlen(kernel_src);

  cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_src, &src_len, &error);
  OPENCL_SUCCEED(error);

  // Construct the actual options string, which involves some varargs
  // magic.
  va_list vl;
  va_start(vl, options_fmt);
  size_t needed = 1 + vsnprintf(NULL, 0, options_fmt, vl);
  char *options = malloc(needed);
  va_start(vl, options_fmt); /* Must re-init. */
  vsnprintf(options, needed, options_fmt, vl);

  // Here we are a bit more generous than usual and do not terminate
  // the process immediately on a build error.  Instead, we print the
  // error messages first.
  error = clBuildProgram(program, 1, &device, options, NULL, NULL);
  if (error != CL_SUCCESS && error != CL_BUILD_PROGRAM_FAILURE) {
    OPENCL_SUCCEED(error);
  }
  free(options);

  cl_build_status build_status;
  OPENCL_SUCCEED(clGetProgramBuildInfo(program,
                                       device,
                                       CL_PROGRAM_BUILD_STATUS,
                                       sizeof(cl_build_status),
                                       &build_status,
                                       NULL));

  if (build_status != CL_SUCCESS) {
    char *build_log;
    size_t ret_val_size;
    OPENCL_SUCCEED(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size));

    build_log = (char*) malloc(ret_val_size+1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);

    // The spec technically does not say whether the build log is
    // zero-terminated, so let's be careful.
    build_log[ret_val_size] = '\0';

    fprintf(stderr, "Build log:\n%s\n", build_log);

    free(build_log);

    OPENCL_SUCCEED(build_status);
  }

  return program;
}
