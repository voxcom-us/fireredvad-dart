import 'dart:math' as math;
import 'dart:typed_data';

import 'constants.dart';
import 'tensor_ops.dart';
import 'types.dart';

/// Streaming VAD inference.
///
/// Processes [feat] (shape [t, dIn]) through the DFSMN network,
/// writing [t] speech probabilities to [probsOut].
void vadInfer(
  VadWeights w,
  VadCaches state,
  Float32List feat,
  int t,
  Float32List probsOut,
  VadWorkspace ws,
) {
  final hidden = ws.hidden;
  final proj = ws.proj;
  final projT = ws.projT;
  final convOut = ws.convOut;
  final fsmnOut = ws.fsmnOut;
  final prevRes = ws.prevRes;
  final tmpTd = ws.tmpTd;
  final projCf = ws.projCf;

  // Input projection: feat[T,80] -> hidden[T,256] -> proj[T,128]
  matmulBiasRelu(feat, w.inpFc1W, w.inpFc1B, hidden, t, dIn, dHidden, true);
  matmulBiasRelu(hidden, w.inpFc2W, w.inpFc2B, proj, t, dHidden, dProj, true);

  // Block 0: FSMN on input projection
  // Transpose proj[T,128] -> projT[128, lookback+T] with cache prepended
  final tIn = lookback + t;
  for (int ch = 0; ch < dProj; ch++) {
    final destOff = ch * tIn;
    // Copy cache
    projT.setRange(destOff, destOff + lookback, state.caches[0][ch]);
    // Copy current proj column
    for (int ti = 0; ti < t; ti++) {
      projT[destOff + lookback + ti] = proj[ti * dProj + ch];
    }
  }

  // Update cache: last LOOKBACK values
  for (int ch = 0; ch < dProj; ch++) {
    final colOff = ch * tIn;
    for (int i = 0; i < lookback; i++) {
      state.caches[0][ch][i] = projT[colOff + t + i];
    }
  }

  // Depthwise conv1d lookback
  depthwiseConv1dLookback(projT, tIn, w.fsmn0Lookback, convOut, t);

  // Residual: proj + conv_out (in channel-first format)
  transposeTd(proj, projCf, t, dProj);
  vecAdd(projCf, convOut, fsmnOut, dProj * t);

  // Transpose back to [T, dProj] for FC layers
  transposeDt(fsmnOut, prevRes, dProj, t);

  // Blocks 1-7
  for (int b = 0; b < 7; b++) {
    final blk = w.blocks[b];

    // FC expansion + projection
    matmulBiasRelu(
        prevRes, blk.fc1W, blk.fc1B, hidden, t, dProj, dHidden, true);
    matmulBiasRelu(hidden, blk.fc2W, null, proj, t, dHidden, dProj, false);

    // Transpose proj[T,128] -> projT[128, lookback+T] with cache
    for (int ch = 0; ch < dProj; ch++) {
      final destOff = ch * tIn;
      projT.setRange(destOff, destOff + lookback, state.caches[b + 1][ch]);
      for (int ti = 0; ti < t; ti++) {
        projT[destOff + lookback + ti] = proj[ti * dProj + ch];
      }
    }

    // Update cache
    for (int ch = 0; ch < dProj; ch++) {
      final colOff = ch * tIn;
      for (int i = 0; i < lookback; i++) {
        state.caches[b + 1][ch][i] = projT[colOff + t + i];
      }
    }

    // Conv1d lookback
    depthwiseConv1dLookback(projT, tIn, blk.lookbackW, convOut, t);

    // Residual: proj + conv_out
    transposeTd(proj, projCf, t, dProj);
    vecAdd(projCf, convOut, fsmnOut, dProj * t);

    // Transpose back + skip connection
    transposeDt(fsmnOut, tmpTd, dProj, t);
    vecAdd(tmpTd, prevRes, prevRes, t * dProj);
  }

  // Output projection: prevRes[T,128] -> hidden[T,256] -> out[T,1] -> sigmoid
  matmulBiasRelu(
      prevRes, w.outFc1W, w.outFc1B, hidden, t, dProj, dHidden, true);
  for (int ti = 0; ti < t; ti++) {
    double sum = w.outFc2B[0];
    final hOff = ti * dHidden;
    for (int m = 0; m < dHidden; m++) {
      sum += hidden[hOff + m] * w.outFc2W[m];
    }
    probsOut[ti] = 1.0 / (1.0 + math.exp(-sum));
  }
}
