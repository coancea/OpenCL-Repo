// Rot-13 for lowercase ASCII.
kernel void rot13(global char *out, global char *in, int n) {
  int gtid = get_global_id(0);
  if (gtid < n) {
    if (in[gtid] >= 'a' && in[gtid] <= 'z') {
      out[gtid] = (in[gtid] - 'a' + 13) % 26 + 'a';
    } else {
      out[gtid] = in[gtid];
    }
  }
}
