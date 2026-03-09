import 'dart:math' as math;
import 'dart:typed_data';

import 'constants.dart';
import 'tensor_ops.dart';
import 'types.dart';

/// Non-streaming AED inference.
///
/// Processes [feat] (shape [t, dIn]) and writes [t * aedNumClasses]
/// probabilities to [probsOut].
void aedInfer(
  AedWeights model,
  Float32List feat,
  int t,
  Float32List probsOut,
  AedWorkspace ws,
) {
  final hidden = ws.hidden;
  final proj = ws.proj;
  final xCf = ws.xCf;
  final convLb = ws.convLb;
  final convLa = ws.convLa;
  final fsmnOut = ws.fsmnOut;
  final prevRes = ws.prevRes;
  final tmpTd = ws.tmpTd;
  final padded = ws.padded;

  final tIn = lookback + t;

  // Input projection
  matmulBiasRelu(
      feat, model.inpFc1W, model.inpFc1B, hidden, t, dIn, dHidden, true);
  matmulBiasRelu(
      hidden, model.inpFc2W, model.inpFc2B, proj, t, dHidden, dProj, true);

  // Block 0: FSMN on input projection
  transposeTd(proj, xCf, t, dProj);

  // Lookback conv: zero-pad left by LOOKBACK
  for (int ch = 0; ch < dProj; ch++) {
    final pOff = ch * tIn;
    padded.fillRange(pOff, pOff + lookback, 0.0);
    final xOff = ch * t;
    padded.setRange(pOff + lookback, pOff + tIn, xCf, xOff);
  }
  depthwiseConv1dLookback(padded, tIn, model.fsmn0Lookback, convLb, t);
  vecAdd(xCf, convLb, fsmnOut, dProj * t);

  // Lookahead conv
  depthwiseConv1dLookahead(xCf, t, model.fsmn0Lookahead, convLa);
  vecAdd(fsmnOut, convLa, fsmnOut, dProj * t);

  transposeDt(fsmnOut, prevRes, dProj, t);

  // Blocks 1-7
  for (int b = 0; b < 7; b++) {
    final blk = model.blocks[b];

    matmulBiasRelu(
        prevRes, blk.fc1W, blk.fc1B, hidden, t, dProj, dHidden, true);
    matmulBiasRelu(hidden, blk.fc2W, null, proj, t, dHidden, dProj, false);

    transposeTd(proj, xCf, t, dProj);

    // Lookback
    for (int ch = 0; ch < dProj; ch++) {
      final pOff = ch * tIn;
      padded.fillRange(pOff, pOff + lookback, 0.0);
      final xOff = ch * t;
      padded.setRange(pOff + lookback, pOff + tIn, xCf, xOff);
    }
    depthwiseConv1dLookback(padded, tIn, blk.lookbackW, convLb, t);
    vecAdd(xCf, convLb, fsmnOut, dProj * t);

    // Lookahead
    depthwiseConv1dLookahead(xCf, t, blk.lookaheadW, convLa);
    vecAdd(fsmnOut, convLa, fsmnOut, dProj * t);

    // Transpose + skip
    transposeDt(fsmnOut, tmpTd, dProj, t);
    vecAdd(tmpTd, prevRes, prevRes, t * dProj);
  }

  // Output: [T,128] -> [T,256] -> [T,3] -> sigmoid
  matmulBiasRelu(
      prevRes, model.outFc1W, model.outFc1B, hidden, t, dProj, dHidden, true);
  matmulBiasRelu(hidden, model.outFc2W, model.outFc2B, probsOut, t, dHidden,
      aedNumClasses, false);
  final total = t * aedNumClasses;
  for (int i = 0; i < total; i++) {
    probsOut[i] = 1.0 / (1.0 + math.exp(-probsOut[i]));
  }
}

/// Classify a segment. Returns best class index and fills [avgProbs] with
/// per-class average probabilities.
int aedClassify(
  AedWeights model,
  Float32List feat,
  int t,
  Float32List avgProbs,
  AedWorkspace ws,
) {
  if (t == 0) return -1;

  aedInfer(model, feat, t, ws.probs, ws);

  for (int c = 0; c < aedNumClasses; c++) {
    avgProbs[c] = 0;
  }
  for (int ti = 0; ti < t; ti++) {
    for (int c = 0; c < aedNumClasses; c++) {
      avgProbs[c] += ws.probs[ti * aedNumClasses + c];
    }
  }

  int best = 0;
  for (int c = 0; c < aedNumClasses; c++) {
    avgProbs[c] /= t;
    if (avgProbs[c] > avgProbs[best]) best = c;
  }
  return best;
}
