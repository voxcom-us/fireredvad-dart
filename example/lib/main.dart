import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:fireredvad/fireredvad.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  runApp(const FireRedVadApp());
}

class FireRedVadApp extends StatelessWidget {
  const FireRedVadApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FireRedVAD Demo',
      theme: ThemeData.dark(useMaterial3: true).copyWith(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepOrange,
          brightness: Brightness.dark,
          surface: Colors.black,
          onSurface: Colors.white,
        ),
        scaffoldBackgroundColor: Colors.black,
        appBarTheme: const AppBarTheme(backgroundColor: Colors.black),
        cardTheme: CardThemeData(color: Colors.grey.shade900),
      ),
      home: const VadDemoPage(),
    );
  }
}

// ---------------------------------------------------------------------------
// Saved speech segment (wraps package SpeechSegment with UI state)
// ---------------------------------------------------------------------------
class SavedSegment {
  final SpeechSegment segment;
  final DateTime timestamp;
  AedResult? aedResult;
  bool isClassifying = false;

  SavedSegment({required this.segment, required this.timestamp});

  VadEvent get event => segment.event;
  Int16List get pcm => segment.pcm;
  double get durationSeconds => segment.durationSeconds;
}

// ---------------------------------------------------------------------------
// WAV builder
// ---------------------------------------------------------------------------
Uint8List buildWav(Int16List pcm) {
  final dataLen = pcm.length * 2;
  final fileLen = 44 + dataLen;
  final wav = ByteData(fileLen);
  const chars = [
    0x52, 0x49, 0x46, 0x46, // RIFF
    0x57, 0x41, 0x56, 0x45, // WAVE
    0x66, 0x6D, 0x74, 0x20, // fmt
    0x64, 0x61, 0x74, 0x61, // data
  ];
  for (int i = 0; i < 4; i++) {
    wav.setUint8(i, chars[i]);
  }
  wav.setUint32(4, fileLen - 8, Endian.little);
  for (int i = 0; i < 4; i++) {
    wav.setUint8(8 + i, chars[4 + i]);
  }
  for (int i = 0; i < 4; i++) {
    wav.setUint8(12 + i, chars[8 + i]);
  }
  wav.setUint32(16, 16, Endian.little);
  wav.setUint16(20, 1, Endian.little); // PCM
  wav.setUint16(22, 1, Endian.little); // mono
  wav.setUint32(24, sampleRate, Endian.little);
  wav.setUint32(28, sampleRate * 2, Endian.little);
  wav.setUint16(32, 2, Endian.little);
  wav.setUint16(34, 16, Endian.little);
  for (int i = 0; i < 4; i++) {
    wav.setUint8(36 + i, chars[12 + i]);
  }
  wav.setUint32(40, dataLen, Endian.little);
  for (int i = 0; i < pcm.length; i++) {
    wav.setInt16(44 + i * 2, pcm[i], Endian.little);
  }
  return wav.buffer.asUint8List();
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------
class VadDemoPage extends StatefulWidget {
  const VadDemoPage({super.key});

  @override
  State<VadDemoPage> createState() => _VadDemoPageState();
}

class _VadDemoPageState extends State<VadDemoPage> {
  FireRedVad? _vad;
  SegmentedVadStream? _stream;
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();
  StreamSubscription<Uint8List>? _audioSub;

  bool _isLoading = true;
  bool _isRecording = false;
  bool _isSpeaking = false;
  String _loadError = '';

  // Waveform data — last N amplitude samples for display
  static const int _waveformLength = 300;
  final List<double> _waveform = List.filled(_waveformLength, 0.0);
  int _waveformHead = 0;

  // Pre-roll duration (configurable via settings)
  static const int _defaultPrerollMs = 300;
  int _prerollMs = _defaultPrerollMs;

  // Saved segments
  final List<SavedSegment> _segments = [];

  // Playback state
  int? _playingIndex;

  // VAD thresholds (defaults)
  double _speechThresholdVal = speechThreshold;
  int _minSpeechFramesVal = minSpeechFrame;
  int _minSilenceFramesVal = minSilenceFrame;
  int _maxSpeechFramesVal = maxSpeechFrame;

  @override
  void initState() {
    super.initState();
    _player.playerStateStream.listen((state) {
      if (state.processingState == ProcessingState.completed) {
        setState(() => _playingIndex = null);
      }
    });
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      final weightsFuture = rootBundle.load('assets/weights.bin');
      final cmvnFuture = rootBundle.loadString('assets/cmvn.json');
      final results = await Future.wait([weightsFuture, cmvnFuture]);

      final vad = FireRedVad.load(
        results[0] as ByteData,
        results[1] as String,
        enableAed: true,
      );

      setState(() {
        _vad = vad;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _loadError = e.toString();
        _isLoading = false;
      });
    }
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      await _stopRecording();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    if (!Platform.isMacOS) {
      final status = await Permission.microphone.request();
      if (!status.isGranted) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Microphone permission denied')),
          );
        }
        return;
      }
    }

    _stream = _vad!.createSegmentedStream(
      speechThreshold: _speechThresholdVal,
      minSpeechFrames: _minSpeechFramesVal,
      minSilenceFrames: _minSilenceFramesVal,
      maxSpeechFrames: _maxSpeechFramesVal,
      prerollMs: _prerollMs,
    );
    _segments.clear();
    _waveform.fillRange(0, _waveformLength, 0.0);
    _waveformHead = 0;

    final audioStream = await _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
        autoGain: false,
        echoCancel: false,
        noiseSuppress: false,
      ),
    );

    _audioSub = audioStream.listen(_onAudioData);

    setState(() {
      _isRecording = true;
      _isSpeaking = false;
    });
  }

  void _onAudioData(Uint8List data) {
    // Copy to aligned buffer — record plugin may deliver odd offsets
    final aligned = Uint8List(data.length)..setAll(0, data);
    final pcm = Int16List.view(aligned.buffer, 0, aligned.length ~/ 2);

    // Update waveform — one bar per ~32ms
    const step = 512; // 512 samples = 32ms at 16kHz
    for (int i = 0; i < pcm.length; i += step) {
      double maxAmp = 0;
      final end = math.min(i + step, pcm.length);
      for (int j = i; j < end; j++) {
        final amp = pcm[j].abs() / 32768.0;
        if (amp > maxAmp) maxAmp = amp;
      }
      _waveform[_waveformHead % _waveformLength] = maxAmp;
      _waveformHead++;
    }

    // Run segmented VAD — returns completed segments
    final segments = _stream!.processChunk(pcm);
    _isSpeaking = _stream!.isSpeaking;

    for (final seg in segments) {
      final saved = SavedSegment(segment: seg, timestamp: DateTime.now());
      _segments.insert(0, saved);
      _classifySegment(saved);
    }

    setState(() {});
  }

  void _classifySegment(SavedSegment saved) {
    if (_vad == null || !_vad!.hasAed) return;
    saved.isClassifying = true;
    final result = _vad!.classifySegment(saved.pcm);
    saved.aedResult = result;
    saved.isClassifying = false;
    if (mounted) setState(() {});
  }

  Future<void> _stopRecording() async {
    await _audioSub?.cancel();
    _audioSub = null;
    await _recorder.stop();

    final trailing = _stream?.flush();
    setState(() {
      _isRecording = false;
      _isSpeaking = false;
      if (trailing != null) {
        final saved = SavedSegment(segment: trailing, timestamp: DateTime.now());
        _segments.insert(0, saved);
        _classifySegment(saved);
      }
    });
  }

  Future<void> _playSegment(int index) async {
    if (_playingIndex == index) {
      // Stop current playback
      await _player.stop();
      setState(() => _playingIndex = null);
      return;
    }

    final segment = _segments[index];
    final wav = buildWav(segment.pcm);

    // Write WAV to temp file
    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/vad_segment_$index.wav');
    await file.writeAsBytes(wav);

    setState(() => _playingIndex = index);
    await _player.setFilePath(file.path);
    _player.play();
  }

  void _showSettingsDialog() {
    var threshold = _speechThresholdVal;
    var minSpeech = _minSpeechFramesVal;
    var minSilence = _minSilenceFramesVal;
    var maxSpeech = _maxSpeechFramesVal;
    var prerollMs = _prerollMs;

    showDialog(
      context: context,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (ctx, setDialogState) {
            return AlertDialog(
              title: const Text('VAD Settings'),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text('Speech Threshold: ${threshold.toStringAsFixed(2)}'),
                    Slider(
                      value: threshold,
                      min: 0.1,
                      max: 0.9,
                      divisions: 80,
                      onChanged: (v) => setDialogState(() => threshold = v),
                    ),
                    const SizedBox(height: 8),
                    Text('Min Speech Frames: $minSpeech'),
                    Slider(
                      value: minSpeech.toDouble(),
                      min: 1,
                      max: 50,
                      divisions: 49,
                      onChanged: (v) =>
                          setDialogState(() => minSpeech = v.round()),
                    ),
                    const SizedBox(height: 8),
                    Text('Min Silence Frames: $minSilence'),
                    Slider(
                      value: minSilence.toDouble(),
                      min: 5,
                      max: 100,
                      divisions: 95,
                      onChanged: (v) =>
                          setDialogState(() => minSilence = v.round()),
                    ),
                    const SizedBox(height: 8),
                    Text('Max Speech Frames: $maxSpeech'),
                    Slider(
                      value: maxSpeech.toDouble(),
                      min: 100,
                      max: 5000,
                      divisions: 49,
                      onChanged: (v) =>
                          setDialogState(() => maxSpeech = v.round()),
                    ),
                    const SizedBox(height: 8),
                    Text('Pre-roll: ${prerollMs}ms'),
                    Slider(
                      value: prerollMs.toDouble(),
                      min: 0,
                      max: 1000,
                      divisions: 20,
                      onChanged: (v) =>
                          setDialogState(() => prerollMs = v.round()),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'At ${framesPerSecond}fps: min speech=${(minSpeech / framesPerSecond * 1000).round()}ms, '
                      'min silence=${(minSilence / framesPerSecond * 1000).round()}ms, '
                      'max speech=${(maxSpeech / framesPerSecond).toStringAsFixed(1)}s',
                      style: TextStyle(fontSize: 12, color: Colors.grey),
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () {
                    setDialogState(() {
                      threshold = speechThreshold;
                      minSpeech = minSpeechFrame;
                      minSilence = minSilenceFrame;
                      maxSpeech = maxSpeechFrame;
                      prerollMs = _defaultPrerollMs;
                    });
                  },
                  child: const Text('Reset'),
                ),
                TextButton(
                  onPressed: () => Navigator.pop(ctx),
                  child: const Text('Cancel'),
                ),
                FilledButton(
                  onPressed: () {
                    setState(() {
                      _speechThresholdVal = threshold;
                      _minSpeechFramesVal = minSpeech;
                      _minSilenceFramesVal = minSilence;
                      _maxSpeechFramesVal = maxSpeech;
                      _prerollMs = prerollMs;
                    });
                    Navigator.pop(ctx);
                    if (_isRecording) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text(
                            'Settings apply on next recording session',
                          ),
                        ),
                      );
                    }
                  },
                  child: const Text('Apply'),
                ),
              ],
            );
          },
        );
      },
    );
  }

  @override
  void dispose() {
    _audioSub?.cancel();
    _recorder.dispose();
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: (!_isLoading && _loadError.isEmpty)
            ? IconButton(
                icon: const Icon(Icons.tune),
                tooltip: 'VAD Settings',
                onPressed: _showSettingsDialog,
              )
            : null,
        title: const Text('FireRedVAD Demo'),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _loadError.isNotEmpty
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  'Failed to load model:\n$_loadError',
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.red),
                ),
              ),
            )
          : Column(
              children: [
                // Waveform + status
                Container(
                  color: _isSpeaking
                      ? Colors.red.withValues(alpha: 0.1)
                      : Colors.white.withValues(alpha: 0.05),
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  child: Column(
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          AnimatedContainer(
                            duration: const Duration(milliseconds: 150),
                            width: 12,
                            height: 12,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: !_isRecording
                                  ? Colors.grey
                                  : _isSpeaking
                                  ? Colors.red
                                  : Colors.green,
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            !_isRecording
                                ? 'Stopped'
                                : _isSpeaking
                                ? 'Speech'
                                : 'Silence',
                            style: Theme.of(context).textTheme.titleSmall
                                ?.copyWith(
                                  color: _isSpeaking
                                      ? Colors.red.shade300
                                      : null,
                                ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      SizedBox(
                        height: 80,
                        width: double.infinity,
                        child: CustomPaint(
                          painter: _BarWaveformPainter(
                            waveform: _waveform,
                            head: _waveformHead,
                            length: _waveformLength,
                            isSpeaking: _isSpeaking,
                            isRecording: _isRecording,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const Divider(height: 1),
                // Segment list
                Expanded(
                  child: _segments.isEmpty
                      ? Center(
                          child: Text(
                            _isRecording
                                ? 'Speak to detect segments...'
                                : 'Segments will appear here',
                            style: TextStyle(color: Colors.grey),
                          ),
                        )
                      : ListView.builder(
                          padding: const EdgeInsets.only(bottom: 80),
                          itemCount: _segments.length,
                          itemBuilder: (context, index) {
                            return _SegmentTile(
                              segment: _segments[index],
                              isPlaying: _playingIndex == index,
                              onPlay: () => _playSegment(index),
                              onClassify: () =>
                                  _classifySegment(_segments[index]),
                            );
                          },
                        ),
                ),
              ],
            ),
      floatingActionButton: _isLoading || _loadError.isNotEmpty
          ? null
          : FloatingActionButton.extended(
              onPressed: _toggleRecording,
              icon: Icon(_isRecording ? Icons.stop : Icons.mic),
              label: Text(_isRecording ? 'Stop' : 'Record'),
              backgroundColor: _isRecording ? Colors.red : null,
              foregroundColor: _isRecording ? Colors.white : null,
            ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}

// ---------------------------------------------------------------------------
// Segment list tile
// ---------------------------------------------------------------------------
class _SegmentTile extends StatelessWidget {
  final SavedSegment segment;
  final bool isPlaying;
  final VoidCallback onPlay;
  final VoidCallback onClassify;

  const _SegmentTile({
    required this.segment,
    required this.isPlaying,
    required this.onPlay,
    required this.onClassify,
  });

  @override
  Widget build(BuildContext context) {
    final evt = segment.event;
    final aed = segment.aedResult;

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.record_voice_over,
                  color: Colors.deepOrange.shade400,
                  size: 20,
                ),
                const SizedBox(width: 8),
                Text(
                  'Segment ${segment.timestamp.hour.toString().padLeft(2, '0')}:'
                  '${segment.timestamp.minute.toString().padLeft(2, '0')}:'
                  '${segment.timestamp.second.toString().padLeft(2, '0')}',
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
                const Spacer(),
                Text(
                  '${segment.durationSeconds.toStringAsFixed(2)}s',
                  style: TextStyle(color: Colors.grey, fontSize: 13),
                ),
                const SizedBox(width: 8),
                // Play button
                SizedBox(
                  width: 36,
                  height: 36,
                  child: IconButton(
                    onPressed: onPlay,
                    icon: Icon(
                      isPlaying ? Icons.stop_circle : Icons.play_circle,
                      color: isPlaying
                          ? Colors.red
                          : Colors.deepOrange.shade400,
                    ),
                    iconSize: 28,
                    padding: EdgeInsets.zero,
                    tooltip: isPlaying ? 'Stop' : 'Play',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              '${evt.startSeconds.toStringAsFixed(2)}s - ${evt.endSeconds.toStringAsFixed(2)}s  '
              '(${segment.pcm.length} samples)',
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
            if (aed != null) ...[
              const SizedBox(height: 8),
              _AedBar(
                label: 'Speech',
                value: aed.probs[0],
                color: Colors.green,
              ),
              _AedBar(label: 'Music', value: aed.probs[1], color: Colors.blue),
              _AedBar(
                label: 'Noise',
                value: aed.probs[2],
                color: Colors.orange,
              ),
            ],
            if (segment.isClassifying)
              const Padding(
                padding: EdgeInsets.only(top: 8),
                child: LinearProgressIndicator(),
              ),
            const SizedBox(height: 8),
            // Mini bar waveform of the segment
            SizedBox(
              height: 32,
              width: double.infinity,
              child: CustomPaint(
                painter: _SegmentBarWaveformPainter(pcm: segment.pcm),
              ),
            ),
            if (aed == null && !segment.isClassifying)
              Align(
                alignment: Alignment.centerRight,
                child: TextButton.icon(
                  onPressed: onClassify,
                  icon: const Icon(Icons.analytics, size: 18),
                  label: const Text('Classify'),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// AED probability bar
// ---------------------------------------------------------------------------
class _AedBar extends StatelessWidget {
  final String label;
  final double value;
  final Color color;

  const _AedBar({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 1),
      child: Row(
        children: [
          SizedBox(
            width: 50,
            child: Text(label, style: const TextStyle(fontSize: 11)),
          ),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(2),
              child: LinearProgressIndicator(
                value: value,
                backgroundColor: Colors.grey.shade800,
                color: color,
                minHeight: 8,
              ),
            ),
          ),
          const SizedBox(width: 6),
          SizedBox(
            width: 40,
            child: Text(
              '${(value * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Live bar waveform painter
// ---------------------------------------------------------------------------
class _BarWaveformPainter extends CustomPainter {
  final List<double> waveform;
  final int head;
  final int length;
  final bool isSpeaking;
  final bool isRecording;

  _BarWaveformPainter({
    required this.waveform,
    required this.head,
    required this.length,
    required this.isSpeaking,
    required this.isRecording,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (!isRecording) return;

    final color = isSpeaking ? Colors.red.shade400 : Colors.green.shade400;
    final midY = size.height / 2;
    final gap = 1.0;
    final totalBarWidth = size.width / length;
    final barWidth = totalBarWidth - gap;
    if (barWidth <= 0) return;

    final paint = Paint()..style = PaintingStyle.fill;

    for (int i = 0; i < length; i++) {
      final idx = ((head - length + i) % length + length) % length;
      final amp = waveform[idx];
      final barHeight = amp * midY;
      final x = i * totalBarWidth;

      // Fade older bars
      final age = (length - i) / length;
      paint.color = color.withValues(alpha: 0.3 + 0.7 * (1 - age));

      // Draw bar centered vertically
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromCenter(
            center: Offset(x + barWidth / 2, midY),
            width: barWidth,
            height: math.max(1, barHeight * 2),
          ),
          const Radius.circular(1),
        ),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _BarWaveformPainter old) => true;
}

// ---------------------------------------------------------------------------
// Segment mini bar waveform painter
// ---------------------------------------------------------------------------
class _SegmentBarWaveformPainter extends CustomPainter {
  final Int16List pcm;

  _SegmentBarWaveformPainter({required this.pcm});

  @override
  void paint(Canvas canvas, Size size) {
    if (pcm.isEmpty) return;

    final paint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.deepOrange.shade300;

    final midY = size.height / 2;
    final numBars = math.min(size.width ~/ 3, 100); // ~3px per bar
    final samplesPerBar = pcm.length ~/ numBars;
    if (samplesPerBar < 1) return;
    final gap = 1.0;
    final totalBarWidth = size.width / numBars;
    final barWidth = totalBarWidth - gap;
    if (barWidth <= 0) return;

    for (int b = 0; b < numBars; b++) {
      final start = b * samplesPerBar;
      final end = math.min(start + samplesPerBar, pcm.length);
      double maxAmp = 0;
      for (int i = start; i < end; i++) {
        final amp = pcm[i].abs() / 32768.0;
        if (amp > maxAmp) maxAmp = amp;
      }
      final barHeight = maxAmp * midY;
      final x = b * totalBarWidth;

      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromCenter(
            center: Offset(x + barWidth / 2, midY),
            width: barWidth,
            height: math.max(1, barHeight * 2),
          ),
          const Radius.circular(1),
        ),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _SegmentBarWaveformPainter old) =>
      old.pcm != pcm;
}
