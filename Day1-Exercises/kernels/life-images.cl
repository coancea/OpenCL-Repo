kernel void life(int n, int m, read_only image2d_t in, write_only image2d_t out) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  // TODO: compute the new liveness and write it to the output image.
}
