typedef int32_t     ElTp;
#define NE          0
#define EPS         0.001

#define lgWARP              5
#define WARP                (1<<lgWARP)

inline uint32_t pred(int32_t k) {
    return (1 - (k & 1));
}
