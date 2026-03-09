import 'dart:typed_data';

import 'aed_inference.dart';
import 'cmvn.dart';
import 'constants.dart' as c;
import 'fbank.dart';
import 'segmented_stream.dart';
import 'state_machine.dart';
import 'types.dart';
import 'vad_inference.dart';
import 'weights.dart' as w;

/// Result from classifying a speech segment with AED.
class AedResult {
  final int bestClass;
  final String label;
  final List<double> probs; // [speech, music, noise]

  AedResult({
    required this.bestClass,
    required this.label,
    required this.probs,
  });

  @override
  String toString() =>
      'AedResult($label: speech=${probs[0].toStringAsFixed(4)}, '
      'music=${probs[1].toStringAsFixed(4)}, '
      'noise=${probs[2].toStringAsFixed(4)})';
}

/// High-level streaming VAD + AED engine.
///
/// Usage:
/// ```dart
/// final vad = FireRedVad.load(weightsData, cmvnJson);
/// final stream = vad.createStream();
///
/// // Feed audio chunks (16-bit PCM, 16kHz, mono)
/// for (final chunk in audioChunks) {
///   final events = stream.processChunk(chunk);
///   for (final event in events) {
///     print(event); // speech_start / speech_end
///   }
/// }
///
/// // Flush at end of stream
/// final finalEvent = stream.flush();
/// ```
class FireRedVad {
  final VadWeights _vadWeights;
  final AedWeights? _aedWeights;
  final Cmvn _cmvn;

  FireRedVad._({
    required VadWeights vadWeights,
    required AedWeights? aedWeights,
    required Cmvn cmvn,
  })  : _vadWeights = vadWeights,
        _aedWeights = aedWeights,
        _cmvn = cmvn;

  /// Load model from binary weight data and CMVN JSON string.
  ///
  /// [weightsData] is the raw bytes of `weights.bin` as a [ByteData].
  /// [cmvnJson] is the contents of `cmvn.json`.
  /// Set [enableAed] to false to skip loading AED weights.
  factory FireRedVad.load(
    ByteData weightsData,
    String cmvnJson, {
    bool enableAed = true,
  }) {
    initFbank();
    final cmvn = loadCmvnFromJson(cmvnJson);
    final weights = w.loadWeights(weightsData, loadAed: enableAed);
    return FireRedVad._(
      vadWeights: weights.vad,
      aedWeights: weights.aed,
      cmvn: cmvn,
    );
  }

  /// Whether AED weights are loaded.
  bool get hasAed => _aedWeights != null;

  /// Create an independent processing stream.
  ///
  /// Each stream has its own caches and state machine, so multiple
  /// streams can process different audio sources concurrently.
  ///
  /// Optional thresholds override the defaults in [VadStateMachine].
  VadStream createStream({
    int chunkMs = 160,
    double? speechThreshold,
    int? minSpeechFrames,
    int? minSilenceFrames,
    int? maxSpeechFrames,
  }) {
    return VadStream._(
      vadWeights: _vadWeights,
      cmvn: _cmvn,
      chunkMs: chunkMs,
      speechThreshold: speechThreshold,
      minSpeechFrames: minSpeechFrames,
      minSilenceFrames: minSilenceFrames,
      maxSpeechFrames: maxSpeechFrames,
    );
  }

  /// Create a segmented stream that emits complete [SpeechSegment]s
  /// with PCM audio data, including a configurable pre-roll buffer.
  ///
  /// This is a higher-level API than [createStream] — instead of raw
  /// speech_start/end events, you get complete audio segments ready for
  /// playback or classification.
  SegmentedVadStream createSegmentedStream({
    int chunkMs = 160,
    int prerollMs = 300,
    double? speechThreshold,
    int? minSpeechFrames,
    int? minSilenceFrames,
    int? maxSpeechFrames,
  }) {
    final inner = createStream(
      chunkMs: chunkMs,
      speechThreshold: speechThreshold,
      minSpeechFrames: minSpeechFrames,
      minSilenceFrames: minSilenceFrames,
      maxSpeechFrames: maxSpeechFrames,
    );
    return SegmentedVadStream(inner: inner, prerollMs: prerollMs);
  }

  /// Classify a speech segment using AED (non-streaming).
  ///
  /// [pcm] should be the PCM samples of the speech segment.
  /// Returns null if AED is not loaded or segment is too short.
  AedResult? classifySegment(Int16List pcm) {
    if (_aedWeights == null) return null;

    final feat = extractFbankSegment(pcm, pcm.length);
    if (feat == null) return null;

    final numFrames = feat.length ~/ c.numMelBins;
    applyCmvn(feat, numFrames, _cmvn);

    final avgProbs = Float32List(c.aedNumClasses);
    final ws = AedWorkspace(numFrames);
    final best = aedClassify(_aedWeights!, feat, numFrames, avgProbs, ws);
    if (best < 0) return null;

    return AedResult(
      bestClass: best,
      label: c.aedLabels[best],
      probs: [avgProbs[0], avgProbs[1], avgProbs[2]],
    );
  }
}

/// A single processing stream with its own caches and state.
class VadStream {
  final VadWeights _vadWeights;
  final Cmvn _cmvn;
  final int _chunkMs;

  final VadCaches _caches = VadCaches();
  late final VadStateMachine _sm;
  late final VadWorkspace _ws;
  late final Int16List _remainder;
  final List<int> _remLen = [0];
  late final Float32List _featBuf;
  late final Float32List _probBuf;
  late final int _maxChunkFrames;

  VadStream._({
    required VadWeights vadWeights,
    required Cmvn cmvn,
    required int chunkMs,
    double? speechThreshold,
    int? minSpeechFrames,
    int? minSilenceFrames,
    int? maxSpeechFrames,
  })  : _vadWeights = vadWeights,
        _cmvn = cmvn,
        _chunkMs = chunkMs {
    final chunkSamples = _chunkMs * c.sampleRate ~/ 1000;
    _maxChunkFrames = chunkSamples ~/ c.frameShift + 2;
    final remainderCap = c.frameLength + chunkSamples;

    _sm = VadStateMachine(
      speechThresholdValue: speechThreshold ?? c.speechThreshold,
      minSpeechFrames: minSpeechFrames ?? c.minSpeechFrame,
      minSilenceFrames: minSilenceFrames ?? c.minSilenceFrame,
      maxSpeechFrames: maxSpeechFrames ?? c.maxSpeechFrame,
    );
    _ws = VadWorkspace(_maxChunkFrames);
    _remainder = Int16List(remainderCap);
    _featBuf = Float32List(_maxChunkFrames * c.numMelBins);
    _probBuf = Float32List(_maxChunkFrames);
  }

  /// Process a chunk of 16-bit PCM audio (16kHz, mono).
  ///
  /// Returns a list of [VadEvent]s detected in this chunk.
  List<VadEvent> processChunk(Int16List pcm) {
    final events = <VadEvent>[];

    final nf = extractFbank(
      pcm,
      pcm.length,
      _remainder,
      _remLen,
      _featBuf,
      _maxChunkFrames,
    );

    if (nf > 0) {
      applyCmvn(_featBuf, nf, _cmvn);
      vadInfer(_vadWeights, _caches, _featBuf, nf, _probBuf, _ws);

      for (int i = 0; i < nf; i++) {
        final evt = _sm.processFrame(_probBuf[i].toDouble());
        if (evt != null) events.add(evt);
      }
    }

    return events;
  }

  /// Flush any pending speech at end of stream.
  VadEvent? flush() => _sm.flush();

  /// Reset stream state for reuse with a new audio source.
  void reset() {
    _caches.reset();
    _sm.reset();
    _remLen[0] = 0;
  }
}
