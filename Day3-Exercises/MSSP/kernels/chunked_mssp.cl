int div_rounding_up(int x, int y) {
  return (x + y - 1) / y;
}

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

kernel void chunked_mssp_stage_one(int n, global int *input, global int4 *output,
                                   local int4 *buf) {
  int gtid = get_global_id(0);
  int ltid = get_local_id(0);
  int gid = get_group_id(0);

  int group_size = get_local_size(0);
  int num_threads = get_global_size(0);

  // How many chunks should each thread process?
  int elems_per_thread = div_rounding_up(n, num_threads);

  int4 carry_in = mssp_mapf(0);
  for (int i = 0; i < elems_per_thread; i++) {
    int j = elems_per_thread * gid * group_size + i * group_size + ltid;

    // TODO: Each thread in the workgroup reads input[j] (if j < n,
    // otherwise neutral element), applies the map function, and then
    // perform a parallel reduction in local memory to reduce this to
    // a single value.  That single value must then be kept as a
    // "carry-in" by a *single* thread for the next iteration of the
    // loop, and fed into the next reduction.  This is a subtle
    // detail!
  }

  // TODO: the first thread writes its result, which is equivalent to
  // the result of the entire workgroup.
}

kernel void chunked_mssp_stage_two(int n, global int4 *input, global int4 *output,
                                   local int4 *buf) {
  // TODO: Just like chunked_mssp_stage_one, except that there is no
  // need to apply the map function to the elements we read.
}
