int intmax(int y, int x) {
  if (y < x) {
    return x;
  } else {
    return y;
  }
}

int4 mssp_mapf(int x) {
  int4 v;
  int x0 = intmax(x, 0);
  v.s0 = x0;
  v.s1 = x0;
  v.s2 = x0;
  v.s3 = 0;
  return v;
}

int4 mssp_redf(int4 x, int4 y) {
  int4 r;
  r.s0 = intmax(intmax(x.s0, y.s0), x.s2 + y.s1);
  r.s1 = intmax(x.s1, x.s3 + y.s1);
  r.s2 = intmax(x.s2 + y.s3, y.s2);
  r.s3 = x.s3 + y.s3;
  return r;
}

kernel void mssp_init(int n, global int *input, global int4 *output) {
  int i = get_global_id(0);

  if (i < n) {
    output[i] = mssp_mapf(input[i]);
  }
}

kernel void tree_mssp(int n, int m, global int4 *input, global int4 *output) {
  int i = get_global_id(0);

  if (i < m) {
    int4 x = input[2*i];
    int4 y = (2*i+1) < n ? input[2*i+1] : mssp_mapf(0);

    output[i] = mssp_redf(x, y);
  }
}
