// This one is not generic in the type, because we need to use a
// special atomic update function.

kernel void atomic_reduction(int n, global int *input, global int *output) {
  int i = get_global_id(0);

  if (i < n) {
    atomic_add(&output[0], input[i]);
  }
}
