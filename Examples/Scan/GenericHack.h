//typedef float     ElTp;
typedef int32_t     ElTp;
#define NE          0
#define EPS         0.001 

#define lgWARP              5
#define WARP                (1<<lgWARP)

#define logWORKGROUP_SIZE   9//10
//#define WORKGROUP_SIZE      (1<<logWORKGROUP_SIZE)
#define WORKGROUP_SIZE      512//128

typedef struct FlgTup {
    uint8_t flg;
    ElTp    val;
} FlgTuple;

ElTp binOp(ElTp v1, ElTp v2) {
    return (v1 + v2);
}

FlgTuple binOpFlg(FlgTuple t1, FlgTuple t2) {
    FlgTuple res;
    if(t2.flg == 0) res.val = binOp(t1.val, t2.val);
    else            res.val = t2.val;
    res.flg = t1.flg | t2.flg;
    return res;
}

//inline ElTp spreadData(float r) { return (ElTp)r; }
ElTp spreadData(float r) { return (ElTp)(r * 20.0); }

