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

kernel void map_mssp(int n, global int *input, global int4 *output) {
  // TODO: compute initial int4 value from each int.
}

kernel void group_mssp(int n, global int4 *input, global int4 *output,
                       local int4 *buf) {
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
      buf[ltid] = mssp_redf(buf[ltid], buf[ltid+active]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // The first thread writes its result, which is equivalent to the
  // result of the entire workgroup.
  if (ltid == 0) {
    output[get_group_id(0)] = buf[0];
  }
}
