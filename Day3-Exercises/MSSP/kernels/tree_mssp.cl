int intmax(int y, int x) {
  if (y < x) {
    return x;
  } else {
    return y;
  }
}

int4 mssp_mapf(int x) {
  int4 v;

  // TODO: fill with initial values.

  return v;
}

int4 mssp_redf(int4 x, int4 y) {
  int4 r;

  // TODO: combine x and y.

  return r;
}

kernel void map_mssp(int n, global int *input, global int4 *output) {
  // TODO: compute initial int4 value from each int.
}

kernel void tree_mssp(int n, int m, global int4 *input, global int4 *output) {
  // TODO: take inspiration from the tree summation kernel.
}
