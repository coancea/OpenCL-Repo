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

kernel void map_mssp(int n, global int *input, global int4 *output) {
  int i = get_global_id(0);

  if (i < n) {
    output[i] = mssp_mapf(input[i]);
  }
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
  for (int skip = 1;
       skip < get_local_size(0);
       skip *= 2) {
    int offset = skip;
    if ((ltid & (2 * skip - 1)) == 0) {
      buf[ltid] = mssp_redf(buf[ltid], buf[ltid+offset]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // The first thread writes its result, which is equivalent to the
  // result of the entire workgroup.
  if (ltid == 0) {
    output[get_group_id(0)] = buf[0];
  }
}
