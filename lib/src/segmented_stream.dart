import 'dart:typed_data';

import 'constants.dart';
import 'fireredvad.dart';
import 'types.dart';

/// A completed speech segment with its PCM audio data.
class SpeechSegment {
  /// The VAD event that ended this segment.
  final VadEvent event;

  /// Raw 16-bit PCM audio (16kHz, mono) including any pre-roll.
  final Int16List pcm;

  /// Duration of the segment in seconds.
  double get durationSeconds => pcm.length / sampleRate;

  SpeechSegment({required this.event, required this.pcm});

  @override
  String toString() =>
      'SpeechSegment(${durationSeconds.toStringAsFixed(2)}s, '
      '${pcm.length} samples)';
}

/// High-level VAD stream that emits complete [SpeechSegment]s with audio data.
///
/// Wraps a [VadStream] and handles PCM accumulation between speech_start/end
/// events, plus a configurable pre-roll buffer to capture speech onset.
///
/// Usage:
/// ```dart
/// final vad = FireRedVad.load(weightsData, cmvnJson);
/// final stream = vad.createSegmentedStream(prerollMs: 300);
///
/// for (final chunk in audioChunks) {
///   final segments = stream.processChunk(chunk);
///   for (final seg in segments) {
///     print('${seg.durationSeconds}s, ${seg.pcm.length} samples');
///   }
/// }
///
/// // Flush trailing speech at end of audio
/// final trailing = stream.flush();
/// ```
class SegmentedVadStream {
  final VadStream _inner;

  // Pre-roll circular buffer
  final int _prerollSamples;
  late Int16List _prerollBuf;
  int _prerollWritePos = 0;
  int _prerollFilled = 0;

  // Segment accumulation
  final List<int> _currentPcm = [];
  bool _accumulating = false;

  /// Creates a segmented stream wrapping the given [VadStream].
  ///
  /// Prefer using [FireRedVad.createSegmentedStream] instead of
  /// constructing directly.
  SegmentedVadStream({
    required VadStream inner,
    int prerollMs = 300,
  })  : _inner = inner,
        _prerollSamples = prerollMs * sampleRate ~/ 1000 {
    _prerollBuf = Int16List(_prerollSamples);
  }

  /// Whether speech is currently being accumulated.
  bool get isSpeaking => _accumulating;

  /// Process a chunk of 16-bit PCM audio (16kHz, mono).
  ///
  /// Returns a list of completed [SpeechSegment]s detected in this chunk.
  /// Most calls return an empty list; a segment is returned when speech ends.
  List<SpeechSegment> processChunk(Int16List pcm) {
    // Run VAD first to get events for this chunk
    final events = _inner.processChunk(pcm);
    final segments = <SpeechSegment>[];
    var chunkAdded = false;

    for (final evt in events) {
      if (evt.type == 'speech_start') {
        _accumulating = true;
        _currentPcm.clear();
        _currentPcm.addAll(_drainPreroll());
        _currentPcm.addAll(pcm);
        chunkAdded = true;
      } else if (evt.type == 'speech_end') {
        if (!chunkAdded) {
          _currentPcm.addAll(pcm);
          chunkAdded = true;
        }
        _accumulating = false;
        if (_currentPcm.isNotEmpty) {
          segments.add(SpeechSegment(
            event: evt,
            pcm: Int16List.fromList(_currentPcm),
          ));
        }
        _currentPcm.clear();
      }
    }

    // After processing events: accumulate or feed preroll
    if (!chunkAdded) {
      if (_accumulating) {
        _currentPcm.addAll(pcm);
      } else {
        _writePreroll(pcm);
      }
    }

    return segments;
  }

  /// Flush any pending speech at end of stream.
  ///
  /// Returns a [SpeechSegment] if speech was in progress, otherwise null.
  SpeechSegment? flush() {
    final evt = _inner.flush();
    if (evt != null && _currentPcm.isNotEmpty) {
      final segment = SpeechSegment(
        event: evt,
        pcm: Int16List.fromList(_currentPcm),
      );
      _accumulating = false;
      _currentPcm.clear();
      return segment;
    }
    _accumulating = false;
    _currentPcm.clear();
    return null;
  }

  /// Reset stream state for reuse with a new audio source.
  void reset() {
    _inner.reset();
    _accumulating = false;
    _currentPcm.clear();
    _prerollBuf = Int16List(_prerollSamples);
    _prerollWritePos = 0;
    _prerollFilled = 0;
  }

  void _writePreroll(Int16List pcm) {
    final cap = _prerollBuf.length;
    if (cap == 0) return;
    for (int i = 0; i < pcm.length; i++) {
      _prerollBuf[_prerollWritePos] = pcm[i];
      _prerollWritePos = (_prerollWritePos + 1) % cap;
      if (_prerollFilled < cap) _prerollFilled++;
    }
  }

  List<int> _drainPreroll() {
    if (_prerollFilled == 0) return const [];
    final cap = _prerollBuf.length;
    final result = List<int>.filled(_prerollFilled, 0);
    final start = (_prerollWritePos - _prerollFilled + cap) % cap;
    for (int i = 0; i < _prerollFilled; i++) {
      result[i] = _prerollBuf[(start + i) % cap];
    }
    _prerollWritePos = 0;
    _prerollFilled = 0;
    return result;
  }
}
