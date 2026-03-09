import 'dart:math' as math;
import 'dart:typed_data';

import 'constants.dart';
import 'types.dart';

/// VAD state machine states.
enum _SmState { silence, possibleSpeech, speech, possibleSilence }

/// Streaming VAD state machine.
///
/// Feeds per-frame speech probabilities and emits [VadEvent]s when
/// speech segments start/end.
class VadStateMachine {
  _SmState _state = _SmState.silence;
  int _frameCnt = 0;
  final Float32List _smoothWindow = Float32List(smoothWindowSize);
  int _smoothLen = 0;
  int _smoothHead = 0;
  double _smoothSum = 0.0;
  int _speechCnt = 0;
  int _silenceCnt = 0;
  bool _hitMaxSpeech = false;
  int _lastSpeechStartFrame = -1;
  int _lastSpeechEndFrame = -1;
  late int _padStartFrame;

  /// Configurable thresholds.
  double speechThresholdValue;
  int minSpeechFrames;
  int minSilenceFrames;
  int maxSpeechFrames;

  VadStateMachine({
    this.speechThresholdValue = speechThreshold,
    this.minSpeechFrames = minSpeechFrame,
    this.minSilenceFrames = minSilenceFrame,
    this.maxSpeechFrames = maxSpeechFrame,
  }) {
    _padStartFrame = math.max(smoothWindowSize, padStartFrame);
  }

  /// Reset to initial state.
  void reset() {
    _state = _SmState.silence;
    _frameCnt = 0;
    _smoothWindow.fillRange(0, smoothWindowSize, 0.0);
    _smoothLen = 0;
    _smoothHead = 0;
    _smoothSum = 0.0;
    _speechCnt = 0;
    _silenceCnt = 0;
    _hitMaxSpeech = false;
    _lastSpeechStartFrame = -1;
    _lastSpeechEndFrame = -1;
  }

  /// Feed one frame's raw probability.
  /// Returns a [VadEvent] if a speech boundary was detected, otherwise null.
  VadEvent? processFrame(double rawProb) {
    _frameCnt++;

    // Smoothing
    if (_smoothLen < smoothWindowSize) {
      _smoothWindow[_smoothLen] = rawProb;
      _smoothSum += rawProb;
      _smoothLen++;
    } else {
      _smoothSum -= _smoothWindow[_smoothHead];
      _smoothWindow[_smoothHead] = rawProb;
      _smoothSum += rawProb;
      _smoothHead = (_smoothHead + 1) % smoothWindowSize;
    }

    final smoothed = _smoothSum / _smoothLen;
    final isSpeech = smoothed >= speechThresholdValue;

    VadEvent? event;

    // Handle max-speech continuation
    if (_hitMaxSpeech) {
      event = VadEvent(
        type: 'speech_start',
        startFrame: _frameCnt,
        endFrame: _frameCnt,
      );
      _lastSpeechStartFrame = _frameCnt;
      _hitMaxSpeech = false;
      // Don't return yet — still need to process the state machine below.
      // But the C code returns after setting has_event=1 and falling through,
      // so we need to also run the switch.
    }

    switch (_state) {
      case _SmState.silence:
        if (isSpeech) {
          _state = _SmState.possibleSpeech;
          _speechCnt = 1;
        } else {
          _silenceCnt++;
          _speechCnt = 0;
        }
        break;

      case _SmState.possibleSpeech:
        if (isSpeech) {
          _speechCnt++;
          if (_speechCnt >= minSpeechFrames) {
            _state = _SmState.speech;
            int start = _frameCnt - _speechCnt + 1 - _padStartFrame;
            if (start < 1) start = 1;
            if (start <= _lastSpeechEndFrame) {
              start = _lastSpeechEndFrame + 1;
            }
            _lastSpeechStartFrame = start;
            _silenceCnt = 0;
            event = VadEvent(
              type: 'speech_start',
              startFrame: start,
              endFrame: _frameCnt,
            );
          }
        } else {
          _state = _SmState.silence;
          _silenceCnt = 1;
          _speechCnt = 0;
        }
        break;

      case _SmState.speech:
        _speechCnt++;
        if (isSpeech) {
          _silenceCnt = 0;
          if (_speechCnt >= maxSpeechFrames) {
            _hitMaxSpeech = true;
            _speechCnt = 0;
            event = VadEvent(
              type: 'speech_end',
              startFrame: _lastSpeechStartFrame,
              endFrame: _frameCnt,
            );
            _lastSpeechEndFrame = _frameCnt;
            _lastSpeechStartFrame = -1;
          }
        } else {
          _state = _SmState.possibleSilence;
          _silenceCnt = 1;
        }
        break;

      case _SmState.possibleSilence:
        _speechCnt++;
        if (isSpeech) {
          _state = _SmState.speech;
          _silenceCnt = 0;
          if (_speechCnt >= maxSpeechFrames) {
            _hitMaxSpeech = true;
            _speechCnt = 0;
            event = VadEvent(
              type: 'speech_end',
              startFrame: _lastSpeechStartFrame,
              endFrame: _frameCnt,
            );
            _lastSpeechEndFrame = _frameCnt;
            _lastSpeechStartFrame = -1;
          }
        } else {
          _silenceCnt++;
          if (_silenceCnt >= minSilenceFrames) {
            _state = _SmState.silence;
            event = VadEvent(
              type: 'speech_end',
              startFrame: _lastSpeechStartFrame,
              endFrame: _frameCnt,
            );
            _lastSpeechEndFrame = _frameCnt;
            _lastSpeechStartFrame = -1;
            _speechCnt = 0;
          }
        }
        break;
    }

    return event;
  }

  /// Flush any pending speech at end of stream.
  VadEvent? flush() {
    if ((_state == _SmState.speech || _state == _SmState.possibleSilence) &&
        _lastSpeechStartFrame >= 0) {
      final evt = VadEvent(
        type: 'speech_end',
        startFrame: _lastSpeechStartFrame,
        endFrame: _frameCnt,
      );
      _state = _SmState.silence;
      _lastSpeechEndFrame = _frameCnt;
      _lastSpeechStartFrame = -1;
      _speechCnt = 0;
      return evt;
    }
    return null;
  }
}
