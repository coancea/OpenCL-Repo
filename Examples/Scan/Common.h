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

#define WORKGROUP_SIZE      512
#define logWORKGROUP_SIZE   9

#endif // COMMON_H
