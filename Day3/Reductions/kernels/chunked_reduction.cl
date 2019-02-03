int div_rounding_up(int x, int y) {
  return (x + y - 1) / y;
}

kernel void chunked_reduction(int n, global elem_t *input, global elem_t *output,
                              local elem_t *buf) {
  int gtid = get_global_id(0);
  int ltid = get_local_id(0);

  int num_threads = get_global_size(0);

  // How many elements should this thread process sequentially?
  int elems_per_thread = div_rounding_up(n, num_threads);

  int x = 0;
  for (int i = 0; i < elems_per_thread; i++) {
    int j = gtid+num_threads*i;
    if (j >= n) {
      break;
    }
    x += input[j];
  }

  buf[ltid] = x;

  barrier(CLK_LOCAL_MEM_FENCE);

  // Then we perform a tree reduction within the workgroup.
  for (int active = get_local_size(0)/2;
       active >= 1;
       active /= 2) {
    if (ltid < active) {
      buf[ltid] = buf[ltid] + buf[ltid+active];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // The first thread writes its result, which is equivalent to the
  // result of the entire workgroup.
  if (ltid == 0) {
    output[get_group_id(0)] = buf[0];
  }
}
