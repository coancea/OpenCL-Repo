#ifndef COMMON_H
#define COMMON_H

#ifndef DO_DEBUG
#define DO_DEBUG 0
#endif

#if DO_DEBUG
#define DEBUG(...) fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG(...) {}
#endif


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MAX(i,j) (((i)>(j)) ? (i) : (j))
#define MIN(i,j) (((i)<(j)) ? (i) : (j))


typedef int           int32_t;
typedef int32_t       ElTp;
typedef unsigned char uint8_t;
typedef unsigned int  uint32_t;


#define lgWARP              5
#define WARP                (1<<lgWARP)


#ifndef WORKGROUP_SIZE
#define WORKGROUP_SIZE      512
#endif

#if (WORKGROUP_SIZE) % 32 != 0
#error "WORKGROUP_SIZE must be a multiple of 32"
#endif

#define logWORKGROUP_SIZE (31 - __builtin_clz((WORKGROUP_SIZE) | 1))

#endif // COMMON_H
