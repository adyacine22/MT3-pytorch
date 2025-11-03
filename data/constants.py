from . import vocabularies

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512
FFT_SIZE = 2048
MEL_LO_HZ = 20.0
MEL_FMIN = 20.0
MEL_FMAX = 8000.0  # Set to Nyquist frequency (sample_rate / 2) to avoid empty filters

PREPEND_ZEROS_WIDTH = 4

# Build codec and vocabulary
VOCAB_CONFIG = vocabularies.VocabularyConfig()
codec = vocabularies.build_codec(VOCAB_CONFIG)
vocab = vocabularies.GenericTokenVocabulary(codec.num_classes)

# Token types
TOKEN_END = vocab.eos_id
TOKEN_START = -1  # This is not used in the new implementation
TOKEN_PAD = 0

# Vocabulary size
VOCAB_SIZE = vocab.vocab_size
