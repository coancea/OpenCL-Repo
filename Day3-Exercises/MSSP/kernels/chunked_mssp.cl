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
  int x0 = intmax(x, 0);
  v.s0 = x0;
  v.s1 = x0;
  v.s2 = x0;
  v.s3 = 0;
  return v;
}

int4 mssp_redf(int4 x, int4 y) {
  int4 r;
  r.s0 = intmax(intmax(x.s0, y.s0), x.s2 + y.s1);
  r.s1 = intmax(x.s1, x.s3 + y.s1);
  r.s2 = intmax(x.s2 + y.s3, y.s2);
  r.s3 = x.s3 + y.s3;
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

    int4 x;
    if (j < n) {
      x = mssp_mapf(input[j]);
    } else {
      x = mssp_mapf(0);
    }

    // First thread in group also handles carry-in.
    if (ltid == 0) {
      x = mssp_redf(carry_in, x);
    }

    buf[ltid] = x;

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

    if (ltid == 0) {
      carry_in = buf[0];
    }
  }

  // The first thread writes its result, which is equivalent to the
  // result of the entire workgroup.
  if (ltid == 0) {
    output[gid] = carry_in;
  }
}

kernel void chunked_mssp_stage_two(int n, global int4 *input, global int4 *output,
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

  if (ltid == 0) {
    output[0] = buf[0];
  }
}
