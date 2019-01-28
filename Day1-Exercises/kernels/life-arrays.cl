// Rules:
// Any live cell with fewer than two live neighbors dies, as if by underpopulation.
// Any live cell with two or three live neighbors lives on to the next generation.
// Any live cell with more than three live neighbors dies, as if by overpopulation.
// Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.

kernel void life(int n, int m, global int *in, global int *out) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  int alive;

  // TODO: compute the new value of 'alive'.

  out[i*n+j] = alive;
}
