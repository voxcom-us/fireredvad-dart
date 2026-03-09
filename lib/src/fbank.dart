import 'dart:math' as math;
import 'dart:typed_data';

import 'constants.dart';

/// Global window and mel filterbank — initialized once via [initFbank].
late final Float32List _window;
late final List<Float32List> _melFb; // [numMelBins][numFftBins]
bool _initialized = false;

double _hertzToMel(double f) => 1127.0 * math.log(1.0 + f / 700.0);
double _melToHertz(double m) => 700.0 * (math.exp(m / 1127.0) - 1.0);

/// Must be called once before any fbank extraction.
void initFbank() {
  if (_initialized) return;

  // Hamming window ^ 0.85
  _window = Float32List(frameLength);
  for (int i = 0; i < frameLength; i++) {
    final hamming =
        0.54 - 0.46 * math.cos(2.0 * math.pi * i / (frameLength - 1));
    _window[i] = math.pow(hamming, 0.85).toDouble();
  }

  // Mel filterbank
  final melLow = _hertzToMel(lowFreq);
  final melHigh = _hertzToMel(highFreq);
  final melPoints = Float64List(numMelBins + 2);
  final binPoints = Float64List(numMelBins + 2);
  for (int i = 0; i < numMelBins + 2; i++) {
    melPoints[i] = melLow + (melHigh - melLow) * i / (numMelBins + 1);
    binPoints[i] = _melToHertz(melPoints[i]) * fftSize / sampleRate;
  }

  _melFb = List.generate(numMelBins, (_) => Float32List(numFftBins));
  for (int m = 0; m < numMelBins; m++) {
    final left = binPoints[m];
    final center = binPoints[m + 1];
    final right = binPoints[m + 2];
    for (int k = 0; k < numFftBins; k++) {
      if (k >= left && k <= center && center > left) {
        _melFb[m][k] = (k - left) / (center - left);
      } else if (k > center && k <= right && right > center) {
        _melFb[m][k] = (right - k) / (right - center);
      }
    }
  }

  _initialized = true;
}

/// In-place radix-2 Cooley-Tukey FFT.
void _fft(Float32List re, Float32List im, int n) {
  // Bit-reversal permutation
  int j = 0;
  for (int i = 0; i < n - 1; i++) {
    if (i < j) {
      double t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
    int m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  // Butterfly
  for (int step = 1; step < n; step <<= 1) {
    final angle = -math.pi / step;
    final wRe = math.cos(angle);
    final wIm = math.sin(angle);
    for (int group = 0; group < n; group += step << 1) {
      double curRe = 1.0, curIm = 0.0;
      for (int pair = 0; pair < step; pair++) {
        final a = group + pair;
        final b = a + step;
        final tRe = curRe * re[b] - curIm * im[b];
        final tIm = curRe * im[b] + curIm * re[b];
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] = re[a] + tRe;
        im[a] = im[a] + tIm;
        final nr = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nr;
      }
    }
  }
}

/// Cached FFT scratch buffers — avoids allocation per frame.
late final Float32List _fftRe = Float32List(fftSize);
late final Float32List _fftIm = Float32List(fftSize);

/// Compute power spectrum of a windowed frame.
void _powerSpectrum(Float32List frame, Float32List power) {
  _fftRe.fillRange(0, fftSize, 0.0);
  _fftIm.fillRange(0, fftSize, 0.0);
  _fftRe.setRange(0, frameLength, frame);
  _fft(_fftRe, _fftIm, fftSize);
  for (int k = 0; k < numFftBins; k++) {
    power[k] = _fftRe[k] * _fftRe[k] + _fftIm[k] * _fftIm[k];
  }
}

/// Streaming fbank extraction.
///
/// Processes [pcm] (16-bit PCM samples), combining with any [remainder] from
/// a prior chunk. Writes mel features to [out] (row-major, [numMelBins] per frame).
/// Returns the number of frames produced.
///
/// [remainder] and [remLen] are updated in-place for the next call.
int extractFbank(
  Int16List pcm,
  int pcmLen,
  Int16List remainder,
  List<int> remLen, // single-element list used as mutable int ref
  Float32List out,
  int maxFrames,
) {
  final total = remLen[0] + pcmLen;

  // Combine remainder + new pcm
  final combined = Int16List(total);
  combined.setRange(0, remLen[0], remainder);
  for (int i = 0; i < pcmLen; i++) {
    combined[remLen[0] + i] = pcm[i];
  }

  int numFrames = 0;
  if (total >= frameLength) {
    numFrames = (total - frameLength) ~/ frameShift + 1;
  }
  if (numFrames > maxFrames) numFrames = maxFrames;

  final windowed = Float32List(frameLength);
  final power = Float32List(numFftBins);

  for (int i = 0; i < numFrames; i++) {
    final start = i * frameShift;
    final prev =
        start > 0 ? combined[start - 1].toDouble() : combined[start].toDouble();
    windowed[0] =
        (combined[start].toDouble() - preEmphasis * prev) * _window[0];
    for (int j = 1; j < frameLength; j++) {
      windowed[j] = (combined[start + j].toDouble() -
              preEmphasis * combined[start + j - 1].toDouble()) *
          _window[j];
    }
    _powerSpectrum(windowed, power);

    final featOffset = i * numMelBins;
    for (int m = 0; m < numMelBins; m++) {
      double sum = 0.0;
      final fb = _melFb[m];
      for (int k = 0; k < numFftBins; k++) {
        sum += fb[k] * power[k];
      }
      out[featOffset + m] = math.log(math.max(sum, 1e-10));
    }
  }

  final consumed = numFrames * frameShift;
  remLen[0] = total - consumed;
  if (remLen[0] > 0) {
    for (int i = 0; i < remLen[0]; i++) {
      remainder[i] = combined[consumed + i];
    }
  }

  return numFrames;
}

/// Non-streaming fbank extraction for a complete segment.
/// Returns feature array of shape [numFrames, numMelBins], or null if too short.
Float32List? extractFbankSegment(Int16List pcm, int pcmLen) {
  int numFrames = 0;
  if (pcmLen >= frameLength) {
    numFrames = (pcmLen - frameLength) ~/ frameShift + 1;
  }
  if (numFrames == 0) return null;

  final feat = Float32List(numFrames * numMelBins);
  final nf = extractFbankSegmentBuf(pcm, pcmLen, feat, numFrames);
  if (nf == 0) return null;
  return feat;
}

/// Non-streaming fbank into pre-allocated buffer. Returns number of frames.
int extractFbankSegmentBuf(
    Int16List pcm, int pcmLen, Float32List out, int maxFrames) {
  int numFrames = 0;
  if (pcmLen >= frameLength) {
    numFrames = (pcmLen - frameLength) ~/ frameShift + 1;
  }
  if (numFrames == 0) return 0;
  if (numFrames > maxFrames) numFrames = maxFrames;

  final windowed = Float32List(frameLength);
  final power = Float32List(numFftBins);

  for (int i = 0; i < numFrames; i++) {
    final start = i * frameShift;
    final prev = start > 0 ? pcm[start - 1].toDouble() : pcm[start].toDouble();
    windowed[0] = (pcm[start].toDouble() - preEmphasis * prev) * _window[0];
    for (int j = 1; j < frameLength; j++) {
      windowed[j] = (pcm[start + j].toDouble() -
              preEmphasis * pcm[start + j - 1].toDouble()) *
          _window[j];
    }
    _powerSpectrum(windowed, power);

    final featOffset = i * numMelBins;
    for (int m = 0; m < numMelBins; m++) {
      double sum = 0.0;
      final fb = _melFb[m];
      for (int k = 0; k < numFftBins; k++) {
        sum += fb[k] * power[k];
      }
      out[featOffset + m] = math.log(math.max(sum, 1e-10));
    }
  }

  return numFrames;
}
