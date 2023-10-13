#ifndef GENERICHACK_H
#define GENERICHACK_H
#include "Common.h"
#define NE          0
#define EPS         0.001 


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

#endif // GENERICHACK_H
