# fireredvad

Streaming Voice Activity Detection (VAD) + Audio Event Detection (AED) using DFSMN. Pure Dart implementation — no native dependencies.

## Features

- Real-time streaming VAD with configurable thresholds
- Audio Event Detection (AED) classifying segments as speech, music, or noise
- Pure Dart — works on iOS, Android, macOS, web, and any Dart platform
- Independent processing streams for concurrent audio sources

## Installation

```yaml
dependencies:
  fireredvad:
    git:
      url: https://github.com/voxcom-us/fireredvad-dart.git
```

## Model Assets

This package requires two model files that must be provided by the consuming application:

- **`weights.bin`** (~4.4 MB) — DFSMN model weights for VAD and AED
- **`cmvn.json`** — Cepstral mean and variance normalization stats

Both files are included in the repository's `assets/` directory. Copy them into your app's assets directory and declare them in your `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/weights.bin
    - assets/cmvn.json
```

## Usage

```dart
import 'package:fireredvad/fireredvad.dart';
import 'package:flutter/services.dart' show rootBundle;

// Load model
final weightsData = await rootBundle.load('assets/weights.bin');
final cmvnJson = await rootBundle.loadString('assets/cmvn.json');

final vad = FireRedVad.load(weightsData, cmvnJson);
final stream = vad.createStream();

// Feed 16-bit PCM audio (16kHz, mono)
for (final chunk in audioChunks) {
  final events = stream.processChunk(chunk);
  for (final event in events) {
    print(event); // VadEvent(speech_start, ...) or VadEvent(speech_end, ...)
  }
}

// Flush at end of audio
final finalEvent = stream.flush();
```

### Custom Thresholds

```dart
final stream = vad.createStream(
  speechThreshold: 0.5,   // default: 0.4
  minSpeechFrames: 10,    // default: 8
  minSilenceFrames: 30,   // default: 20
  maxSpeechFrames: 3000,  // default: 2000
);
```

### Audio Event Detection (AED)

Classify a speech segment as speech, music, or noise:

```dart
final vad = FireRedVad.load(weightsData, cmvnJson, enableAed: true);

// After detecting a speech segment, classify it
final result = vad.classifySegment(segmentPcm);
if (result != null) {
  print(result.label);  // "speech", "music", or "noise"
  print(result.probs);  // [speechProb, musicProb, noiseProb]
}
```

## Example

See the [example/](example/) directory for a Flutter demo app with:

- Real-time microphone recording with live waveform
- Speech segment detection and saving
- AED classification of detected segments
- Configurable VAD thresholds

```sh
cd example
flutter run
```

## Audio Format

All audio input must be **16-bit signed PCM, 16kHz sample rate, mono channel**.

## License

See [LICENSE](LICENSE) for details.
