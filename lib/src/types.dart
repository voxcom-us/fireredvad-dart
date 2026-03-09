import 'dart:typed_data';

import 'constants.dart';

/// CMVN normalization stats.
class Cmvn {
  final Float32List means;
  final Float32List invStd;

  Cmvn({required this.means, required this.invStd});
}

/// Single FSMN block weights (VAD — lookback only).
class FsmnBlock {
  final Float32List fc1W; // [dProj * dHidden]
  final Float32List fc1B; // [dHidden]
  final Float32List fc2W; // [dHidden * dProj]
  final Float32List lookbackW; // [dProj * dFilter]

  FsmnBlock({
    required this.fc1W,
    required this.fc1B,
    required this.fc2W,
    required this.lookbackW,
  });
}

/// VAD model weights (~2.3 MB).
class VadWeights {
  final Float32List inpFc1W; // [dIn * dHidden]
  final Float32List inpFc1B; // [dHidden]
  final Float32List inpFc2W; // [dHidden * dProj]
  final Float32List inpFc2B; // [dProj]
  final Float32List fsmn0Lookback; // [dProj * dFilter]
  final List<FsmnBlock> blocks; // 7 blocks
  final Float32List outFc1W; // [dProj * dHidden]
  final Float32List outFc1B; // [dHidden]
  final Float32List outFc2W; // [dHidden * 1]
  final Float32List outFc2B; // [1]

  VadWeights({
    required this.inpFc1W,
    required this.inpFc1B,
    required this.inpFc2W,
    required this.inpFc2B,
    required this.fsmn0Lookback,
    required this.blocks,
    required this.outFc1W,
    required this.outFc1B,
    required this.outFc2W,
    required this.outFc2B,
  });
}

/// Single FSMN block weights (AED — lookback + lookahead).
class AedFsmnBlock {
  final Float32List fc1W;
  final Float32List fc1B;
  final Float32List fc2W;
  final Float32List lookbackW;
  final Float32List lookaheadW;

  AedFsmnBlock({
    required this.fc1W,
    required this.fc1B,
    required this.fc2W,
    required this.lookbackW,
    required this.lookaheadW,
  });
}

/// AED model weights (~2.3 MB).
class AedWeights {
  final Float32List inpFc1W;
  final Float32List inpFc1B;
  final Float32List inpFc2W;
  final Float32List inpFc2B;
  final Float32List fsmn0Lookback;
  final Float32List fsmn0Lookahead;
  final List<AedFsmnBlock> blocks; // 7 blocks
  final Float32List outFc1W;
  final Float32List outFc1B;
  final Float32List outFc2W; // [dHidden * aedNumClasses]
  final Float32List outFc2B; // [aedNumClasses]

  AedWeights({
    required this.inpFc1W,
    required this.inpFc1B,
    required this.inpFc2W,
    required this.inpFc2B,
    required this.fsmn0Lookback,
    required this.fsmn0Lookahead,
    required this.blocks,
    required this.outFc1W,
    required this.outFc1B,
    required this.outFc2W,
    required this.outFc2B,
  });
}

/// Per-stream mutable caches for streaming VAD.
class VadCaches {
  /// caches[block][channel][lookback_position]
  final List<List<Float32List>> caches;

  VadCaches()
      : caches = List.generate(
          nBlocks,
          (_) => List.generate(
            dProj,
            (_) => Float32List(lookback),
          ),
        );

  void reset() {
    for (final block in caches) {
      for (final ch in block) {
        ch.fillRange(0, ch.length, 0.0);
      }
    }
  }
}

/// Pre-allocated scratch buffers for VAD inference.
class VadWorkspace {
  final int maxT;
  final Float32List hidden;
  final Float32List proj;
  final Float32List projT;
  final Float32List convOut;
  final Float32List fsmnOut;
  final Float32List prevRes;
  final Float32List tmpTd;
  final Float32List projCf;

  VadWorkspace(this.maxT)
      : hidden = Float32List(maxT * dHidden),
        proj = Float32List(maxT * dProj),
        projT = Float32List(dProj * (lookback + maxT)),
        convOut = Float32List(dProj * maxT),
        fsmnOut = Float32List(dProj * maxT),
        prevRes = Float32List(maxT * dProj),
        tmpTd = Float32List(maxT * dProj),
        projCf = Float32List(dProj * maxT);
}

/// Pre-allocated scratch buffers for AED inference.
class AedWorkspace {
  final int maxT;
  final Float32List hidden;
  final Float32List proj;
  final Float32List xCf;
  final Float32List convLb;
  final Float32List convLa;
  final Float32List fsmnOut;
  final Float32List prevRes;
  final Float32List tmpTd;
  final Float32List padded;
  final Float32List feat;
  final Float32List probs;

  AedWorkspace(this.maxT)
      : hidden = Float32List(maxT * dHidden),
        proj = Float32List(maxT * dProj),
        xCf = Float32List(dProj * maxT),
        convLb = Float32List(dProj * maxT),
        convLa = Float32List(dProj * maxT),
        fsmnOut = Float32List(dProj * maxT),
        prevRes = Float32List(maxT * dProj),
        tmpTd = Float32List(maxT * dProj),
        padded = Float32List(dProj * (lookback + maxT)),
        feat = Float32List(maxT * numMelBins),
        probs = Float32List(maxT * aedNumClasses);
}

/// VAD event emitted by the state machine.
class VadEvent {
  final String type; // 'speech_start' or 'speech_end'
  final int startFrame;
  final int endFrame;

  VadEvent({
    required this.type,
    required this.startFrame,
    required this.endFrame,
  });

  double get startSeconds => (startFrame - 1) / framesPerSecond;
  double get endSeconds => endFrame / framesPerSecond;

  @override
  String toString() =>
      'VadEvent($type, ${startSeconds.toStringAsFixed(3)}s - ${endSeconds.toStringAsFixed(3)}s)';
}
