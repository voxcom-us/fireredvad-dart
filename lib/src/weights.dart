import 'dart:typed_data';

import 'constants.dart';
import 'types.dart';

/// Load VAD and AED weights from a binary blob (e.g. from an asset bundle).
///
/// The binary format is: 4-byte magic "FRVD", 4-byte version, then
/// sequential float32 arrays for VAD weights then AED weights.
///
/// If [loadAed] is false, only VAD weights are loaded and aed will be null.
({VadWeights vad, AedWeights? aed}) loadWeights(
  ByteData data, {
  bool loadAed = true,
}) {
  int offset = 0;

  // Check magic
  final magic = String.fromCharCodes([
    data.getUint8(0),
    data.getUint8(1),
    data.getUint8(2),
    data.getUint8(3),
  ]);
  if (magic != 'FRVD') {
    throw FormatException('Bad magic in weights: "$magic"');
  }
  offset += 4;

  // Version
  // final version = data.getUint32(offset, Endian.little);
  offset += 4;

  Float32List readFloats(int count) {
    final list = Float32List(count);
    for (int i = 0; i < count; i++) {
      list[i] = data.getFloat32(offset, Endian.little);
      offset += 4;
    }
    return list;
  }

  // VAD weights
  final inpFc1W = readFloats(dIn * dHidden);
  final inpFc1B = readFloats(dHidden);
  final inpFc2W = readFloats(dHidden * dProj);
  final inpFc2B = readFloats(dProj);
  final fsmn0Lookback = readFloats(dProj * dFilter);

  final blocks = <FsmnBlock>[];
  for (int i = 0; i < 7; i++) {
    blocks.add(FsmnBlock(
      fc1W: readFloats(dProj * dHidden),
      fc1B: readFloats(dHidden),
      fc2W: readFloats(dHidden * dProj),
      lookbackW: readFloats(dProj * dFilter),
    ));
  }

  final outFc1W = readFloats(dProj * dHidden);
  final outFc1B = readFloats(dHidden);
  final outFc2W = readFloats(dHidden * 1);
  final outFc2B = readFloats(1);

  final vad = VadWeights(
    inpFc1W: inpFc1W,
    inpFc1B: inpFc1B,
    inpFc2W: inpFc2W,
    inpFc2B: inpFc2B,
    fsmn0Lookback: fsmn0Lookback,
    blocks: blocks,
    outFc1W: outFc1W,
    outFc1B: outFc1B,
    outFc2W: outFc2W,
    outFc2B: outFc2B,
  );

  AedWeights? aed;
  if (loadAed) {
    final aInpFc1W = readFloats(dIn * dHidden);
    final aInpFc1B = readFloats(dHidden);
    final aInpFc2W = readFloats(dHidden * dProj);
    final aInpFc2B = readFloats(dProj);
    final aFsmn0Lookback = readFloats(dProj * dFilter);
    final aFsmn0Lookahead = readFloats(dProj * dFilter);

    final aBlocks = <AedFsmnBlock>[];
    for (int i = 0; i < 7; i++) {
      aBlocks.add(AedFsmnBlock(
        fc1W: readFloats(dProj * dHidden),
        fc1B: readFloats(dHidden),
        fc2W: readFloats(dHidden * dProj),
        lookbackW: readFloats(dProj * dFilter),
        lookaheadW: readFloats(dProj * dFilter),
      ));
    }

    final aOutFc1W = readFloats(dProj * dHidden);
    final aOutFc1B = readFloats(dHidden);
    final aOutFc2W = readFloats(dHidden * aedNumClasses);
    final aOutFc2B = readFloats(aedNumClasses);

    aed = AedWeights(
      inpFc1W: aInpFc1W,
      inpFc1B: aInpFc1B,
      inpFc2W: aInpFc2W,
      inpFc2B: aInpFc2B,
      fsmn0Lookback: aFsmn0Lookback,
      fsmn0Lookahead: aFsmn0Lookahead,
      blocks: aBlocks,
      outFc1W: aOutFc1W,
      outFc1B: aOutFc1B,
      outFc2W: aOutFc2W,
      outFc2B: aOutFc2B,
    );
  }

  return (vad: vad, aed: aed);
}
