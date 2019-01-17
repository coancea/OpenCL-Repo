#ifndef TRANSP_BRIDGE
#define TRANSP_BRIDGE

typedef float real;

real arithmFun(real accum, real a) {
    return (a*a - accum);  // sqrt(accum) + a*a;
}

#endif //TRANSP_BRIDGE
