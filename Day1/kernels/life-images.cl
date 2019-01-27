const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;

int get(int n, int m, read_only image2d_t in, int i, int j) {
  if (i >= n || i < 0 || j >= m || j < 0) {
    return 0;
  } else {
    uint4 v = read_imageui(in, sampler, (int2)(i,j));
    return v.s0;
  }
}

kernel void life(int n, int m, read_only image2d_t in, write_only image2d_t out) {
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

  write_imageui(out, (int2)(i,j), (uint4)(alive,0,0,0));
}
