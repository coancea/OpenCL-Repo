int get(int n, int m, global int *in, int i, int j) {
  if (i >= n || i < 0 || j >= m || j < 0) {
    return 0;
  } else {
    return in[i*m+j];
  }
}

kernel void life(int n, int m, global int *in, global int *out) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  if (i >= n || j >= m) {
    return;
  }

  int neighbours =
    get(n, m, in, i-1, j-1) + get(n, m, in, i-1, j) + get(n, m, in, i-1, j+1) +
    get(n, m, in, i, j-1) + get(n, m, in, i, j+1) +
    get(n, m, in, i+1, j-1) + get(n, m, in, i+1, j) + get(n, m, in, i+1, j+1);

  int alive = get(n, m, in, i, j);

  if (alive) {
    if (neighbours == 2 || neighbours == 3) {
      alive = 1;
    } else {
      alive = 0;
    }
  } else {
    if (neighbours == 3) {
      alive = 1;
    } else {
      alive = 0;
    }
  }

  out[i*n+j] = alive;
}
