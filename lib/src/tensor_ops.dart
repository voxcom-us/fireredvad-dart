import 'dart:typed_data';

import 'constants.dart';

/// MatMul: C[T,N] = A[T,M] * B[M,N] + bias[N], with optional ReLU.
///
/// Uses ikj loop order for cache-friendly access, with k-unrolling by 4.
void matmulBiasRelu(
  Float32List a,
  Float32List b,
  Float32List? bias,
  Float32List c,
  int t,
  int m,
  int n,
  bool doRelu,
) {
  // Initialize C with bias or zero
  for (int i = 0; i < t; i++) {
    final cOff = i * n;
    if (bias != null) {
      c.setRange(cOff, cOff + n, bias);
    } else {
      c.fillRange(cOff, cOff + n, 0.0);
    }
  }

  // ikj accumulation with k-unrolling by 4
  for (int i = 0; i < t; i++) {
    final cOff = i * n;
    final aOff = i * m;
    int k = 0;
    for (; k + 3 < m; k += 4) {
      final a0 = a[aOff + k];
      final a1 = a[aOff + k + 1];
      final a2 = a[aOff + k + 2];
      final a3 = a[aOff + k + 3];
      final b0Off = k * n;
      final b1Off = (k + 1) * n;
      final b2Off = (k + 2) * n;
      final b3Off = (k + 3) * n;
      for (int j = 0; j < n; j++) {
        c[cOff + j] += a0 * b[b0Off + j] +
            a1 * b[b1Off + j] +
            a2 * b[b2Off + j] +
            a3 * b[b3Off + j];
      }
    }
    for (; k < m; k++) {
      final aIk = a[aOff + k];
      final bOff = k * n;
      for (int j = 0; j < n; j++) {
        c[cOff + j] += aIk * b[bOff + j];
      }
    }
  }

  // ReLU
  if (doRelu) {
    final total = t * n;
    for (int i = 0; i < total; i++) {
      if (c[i] < 0.0) c[i] = 0.0;
    }
  }
}

/// Depthwise Conv1d lookback (causal).
/// Input:  x[dProj][tIn] where tIn = lookback + T
/// Filter: w[dProj][dFilter]
/// Output: out[dProj][T]
void depthwiseConv1dLookback(
  Float32List x,
  int tIn,
  Float32List w,
  Float32List out,
  int t,
) {
  for (int ch = 0; ch < dProj; ch++) {
    final xChOff = ch * tIn;
    final wChOff = ch * dFilter;
    final oChOff = ch * t;
    for (int ti = 0; ti < t; ti++) {
      double sum = 0.0;
      final xp = xChOff + ti;
      for (int k = 0; k < dFilter; k++) {
        sum += w[wChOff + k] * x[xp + k];
      }
      out[oChOff + ti] = sum;
    }
  }
}

/// Depthwise Conv1d lookahead (non-causal, for AED).
/// Input:  x[dProj][T]
/// Filter: w[dProj][dFilter]
/// Output: out[dProj][T]
void depthwiseConv1dLookahead(
  Float32List x,
  int t,
  Float32List w,
  Float32List out,
) {
  for (int ch = 0; ch < dProj; ch++) {
    final xChOff = ch * t;
    final wChOff = ch * dFilter;
    final oChOff = ch * t;
    for (int ti = 0; ti < t; ti++) {
      double sum = 0.0;
      for (int k = 0; k < dFilter; k++) {
        final idx = ti + 1 + k;
        if (idx < t) {
          sum += w[wChOff + k] * x[xChOff + idx];
        }
      }
      out[oChOff + ti] = sum;
    }
  }
}

/// Transpose [T, D] -> [D, T].
void transposeTd(Float32List input, Float32List output, int t, int d) {
  for (int ti = 0; ti < t; ti++) {
    for (int di = 0; di < d; di++) {
      output[di * t + ti] = input[ti * d + di];
    }
  }
}

/// Transpose [D, T] -> [T, D].
void transposeDt(Float32List input, Float32List output, int d, int t) {
  for (int di = 0; di < d; di++) {
    for (int ti = 0; ti < t; ti++) {
      output[ti * d + di] = input[di * t + ti];
    }
  }
}

/// Element-wise addition: out[i] = a[i] + b[i].
void vecAdd(Float32List a, Float32List b, Float32List out, int n) {
  for (int i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}
