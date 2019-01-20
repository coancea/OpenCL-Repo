kernel void tree_reduction(int n, int m, global elem_t *input, global elem_t *output) {
  int i = get_global_id(0);

  if (i < m) {
    int x = input[2*i];
    int y = (2*i+1) < n ? input[2*i+1] : 0;

    output[i] = x + y;
  }
}
