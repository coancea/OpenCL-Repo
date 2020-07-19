kernel void group_reduction(int n, global elem_t *input, global elem_t *output,
                            local elem_t *buf) {
  int gtid = get_global_id(0);
  int ltid = get_local_id(0);

  // Every thread fetches either an element of the input (if in
  // bounds), or zero (if out of bounds).
  buf[ltid] = gtid < n ? input[gtid] : 0;

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
