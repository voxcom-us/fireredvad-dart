/// Audio constants
const int sampleRate = 16000;
const int frameLength = 400;
const int frameShift = 160;
const int fftSize = 512;
const int numFftBins = fftSize ~/ 2 + 1; // 257
const int numMelBins = 80;
const int framesPerSecond = 100;

/// DFSMN dimensions
const int dIn = 80;
const int dHidden = 256;
const int dProj = 128;
const int dFilter = 20;
const int nBlocks = 8;
const int lookback = 19; // LOOKBACK = D_FILTER - 1

/// AED
const int aedNumClasses = 3;
const List<String> aedLabels = ['speech', 'music', 'noise'];

/// VAD state machine defaults
const int smoothWindowSize = 5;
const double speechThreshold = 0.4;
const int padStartFrame = 5;
const int minSpeechFrame = 8;
const int maxSpeechFrame = 2000;
const int minSilenceFrame = 20;

/// Fbank
const double lowFreq = 20.0;
const double highFreq = 8000.0;
const double preEmphasis = 0.97;
