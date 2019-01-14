float fib(int n) {
  float x = 1, y = 1;

  for (int i = 0; i < n; i++) {
    float z = x + y;
    x = y;
    y = z;
  }

  return x;
}

float fact(int n) {
  float x = 1;

  for (int i = 2; i <= n; i++) {
    x *= i;
  }

  return x;
}

kernel void fibfact(int k, global float *out, global int *ns, global int *op) {
  int gtid = get_global_id(0);
  if (gtid < k) {
    int n = ns[gtid];
    int x;
    if (op[gtid] == 1) {
      x = fib(n);
    } else {
      // Do fact(n)
      x = fact(n);
    }
    out[gtid] = x;
  }
}
