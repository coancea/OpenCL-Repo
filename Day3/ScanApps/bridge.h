typedef int32_t     ElTp;
#define NE          0
#define EPS         0.001

#define lgWARP              5
#define WARP                (1<<lgWARP)

typedef struct FlgTup {
    uint8_t flg;
    ElTp    val;
} FlgTuple;

inline ElTp binOp(ElTp v1, ElTp v2) {
    return (v1 + v2);
}

inline FlgTuple binOpFlg(FlgTuple t1, FlgTuple t2) {
    FlgTuple res;
    if(t2.flg == 0) res.val = binOp(t1.val, t2.val);
    else            res.val = t2.val;
    res.flg = t1.flg | t2.flg;
    return res;
}

inline uint32_t pred(int32_t k) {
//    int32_t kk = k;
//    if (k < 0) { kk = 0 - k; }
//    return (1 - (kk % 2));
    return (1 - (k & 1));
}
