/// Streaming Voice Activity Detection + Audio Event Detection
/// using DFSMN (Deep Feed-forward Sequential Memory Network).
///
/// Pure Dart implementation — no native dependencies.
library fireredvad;

export 'src/constants.dart'
    show
        sampleRate,
        frameShift,
        framesPerSecond,
        aedNumClasses,
        aedLabels,
        speechThreshold,
        minSpeechFrame,
        minSilenceFrame,
        maxSpeechFrame;
export 'src/fireredvad.dart' show FireRedVad, VadStream, AedResult;
export 'src/segmented_stream.dart' show SegmentedVadStream, SpeechSegment;
export 'src/types.dart' show VadEvent, Cmvn;
export 'src/state_machine.dart' show VadStateMachine;
