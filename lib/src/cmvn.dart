import 'dart:convert';
import 'dart:typed_data';

import 'constants.dart';
import 'types.dart';

/// Load CMVN stats from a JSON string.
///
/// Expected format: `{"means": [...], "inv_std": [...]}`
Cmvn loadCmvnFromJson(String jsonStr) {
  final map = json.decode(jsonStr) as Map<String, dynamic>;

  final meansList = (map['means'] as List).cast<num>();
  final invStdList = (map['inv_std'] as List).cast<num>();

  if (meansList.length != numMelBins || invStdList.length != numMelBins) {
    throw ArgumentError(
        'CMVN expected $numMelBins values, got means=${meansList.length}, inv_std=${invStdList.length}');
  }

  final means = Float32List(numMelBins);
  final invStd = Float32List(numMelBins);
  for (int i = 0; i < numMelBins; i++) {
    means[i] = meansList[i].toDouble();
    invStd[i] = invStdList[i].toDouble();
  }

  return Cmvn(means: means, invStd: invStd);
}

/// Apply CMVN normalization in-place to feat[T][numMelBins].
void applyCmvn(Float32List feat, int t, Cmvn cmvn) {
  for (int ti = 0; ti < t; ti++) {
    final off = ti * numMelBins;
    for (int d = 0; d < numMelBins; d++) {
      feat[off + d] = (feat[off + d] - cmvn.means[d]) * cmvn.invStd[d];
    }
  }
}
