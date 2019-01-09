// Rot-13 for lowercase ASCII.
kernel void rot13(global char *out, global char *in, int n) {
  int i = get_global_id(0);
  if (i < n) {
    if (in[i] >= 'a' && in[i] <= 'z') {
      out[i] = (in[i] - 'a' + 13) % 26 + 'a';
    } else {
      out[i] = in[i];
    }
  }
}
