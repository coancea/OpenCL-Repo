kernel void sum_rows(int n, global int *output, global int *input) {
  int row = get_global_id(0);
  if (row < n) {
    int sum = 0;
    for (int col = 0; col < n; col++) {
      sum += input[row*n+col];
    }
    output[row] = sum;
  }
}

kernel void sum_cols(int n, global int *output, global int *input) {
  int col = get_global_id(0);
  if (col < n) {
    int sum = 0;
    for (int row = 0; row < n; row++) {
      sum += input[row*n+col];
    }
    output[col] = sum;
  }
}
