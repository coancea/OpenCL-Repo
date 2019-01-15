// Naive, un-optimised matrix multiplication.
//
// Assumes all matrices are stored in row-major form.
//
// Assumes that a type elem_t is defined with -Delem_t=foo.
//
// Defined as a two-dimensional kernel, where thread (i,j) produces
// element (i,j) of the output matrix.

kernel void matmul(int n, int m, int k,
                   global elem_t *out,
                   global elem_t *xss,
                   global elem_t *yss) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  if (i < n && j < k) {
    elem_t res = 0;

    for (int l = 0; l < m; l++) {
      res += xss[i * m + l] * xss[l * k + j];
    }

    out[i * k + j] = res;
  }
}
