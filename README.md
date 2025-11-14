# MT3-PyTorch

PyTorch implementation of [MT3: Sequence-to-Sequence Piano Transcription with Transformers](https://arxiv.org/abs/2107.09142).

---

## Project Description

**MT3-PyTorch** is a multi-track music transcription system that converts polyphonic audio into MIDI using a T5 transformer architecture. It combines:
- **Spectrogram Encoder**: Converts 16kHz audio → 512 mel-spectrogram features
- **T5 Transformer**: Encoder-Decoder (6L+6L, 8 heads, dim=512)
- **MIDI Decoder**: Generates note sequences via vocabulary decoding

Trained on MAESTRO (piano) and SLAKH2100 (multi-instrument) datasets.

---

## Quick Setup

### 1. Clone & Create Environment
```bash
git clone <repository-url> MT3-pytorch 
cd MT3-pytorch
conda create -n mt3-pytorch python=3.11 
conda activate mt3-pytorch
```

### 2. Install via pyproject.toml

**Choose your setup:**

#### **macOS (Apple Silicon / Intel)**
```bash
# PyTorch with Metal Performance Shaders (MPS) support
pip install -e .[dev]

# Verify GPU access
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

#### **Linux (CUDA GPU)**
```bash
# Install PyTorch with CUDA 12.6
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126


# Install project + dev tools
pip install -e .[dev]

# Verify GPU access
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name()}')"
```

#### **Linux (CPU only)**
```bash
pip install -e .[dev]
```

**Installation profiles** (all include torch 2.8.0 with CPU support):
- `pip install -e .` → Core only
- `pip install -e .[dev]` → + pytest, black, mypy (recommended)
- `pip install -e .[datasets]` → + tfrecord, pydub
- `pip install -e .[notebook]` → + jupyter, matplotlib
- `pip install -e .[all]` → Everything

> **Note**: `pyproject.toml` pins PyTorch 2.8.0. For CUDA, reinstall after initial setup with the appropriate index URL above.

---

## Dataset Links

| Dataset | Size | Split | Download |
|---------|------|-------|----------|
| **MAESTRO v2.0.0** | 101GB (120GB uncompressed) | train/valid/test | [Download website](https://magenta.withgoogle.com/datasets/maestro#v300) |
| **SLAKH2100** | ~100GB| train/valid/test | [Download website](https://zenodo.org/records/4599666) |

**Setup:**
```bash
# Extract to datasets/ directory (created if missing)
# Structure: datasets/maestro-v3.0.0/ and datasets/slakh2100_flac_redux/
# Auto-detected by config/data_config.py
```

---

## Testing

```bash
# Quick smoke test
pytest tests/test_vocabularies.py tests/test_layers.py -v

# Full suite (10 files, 53 tests, ~11s)
pytest -v

# With coverage report
pytest --cov --cov-report=html
# Open htmlcov/index.html
```

Test categories:
- Unit tests (vocabularies, layers, preprocessing)
- Integration tests (dataset loading, model I/O)
- Device tests (CPU, CUDA, MPS support)

### Chunk dataset profiling

Use the performance harnesses in `tests/` to sanity check data IO and tokenization:

```bash
# Profile tokenization cost while sweeping max_examples_per_mix
pytest tests/test_chunk_tokenization_profile.py -v

# Profile DataLoader throughput
pytest tests/test_chunk_dataloader_profile.py -v
```

Both tests read the default manifest from `cache/chunk_manifest.parquet`. Override paths or knobs with env vars, e.g.:

```bash
export CHUNK_TOKEN_PROFILE_MANIFEST=datasets/custom_manifest.parquet
export CHUNK_TOKEN_PROFILE_MIX_VALUES="1,2,4"
export CHUNK_TOKEN_PROFILE_SAMPLES=32
export CHUNK_TOKEN_PROFILE_DATASET='{"feature_type": "waveform", "load_tokens": true}'
pytest tests/test_chunk_tokenization_profile.py -v
```

Each run writes JSON logs under `test_files/test_chunk_tokenization_profile/` for later comparison.

---

## Training

### Configuration Files
- `config/training_config.py` → Hyperparameters, batch size, warmup, validation freq
- `config/T5config.py` → Model architecture (embedding dim, heads, layers)
- `config/data_config.py` → Dataset paths, audio params (sample rate, FFT, mel bins)

### Commands

**Training commands** (MAESTRO + SLAKH):
```bash
# Test configuration (small, fast, debugging)
python training/train_multitrack.py --config multitrack_test

# Full production training
python training/train_multitrack.py --config multitrack_full

# A100 training : 
python training/train_multitrack.py --config multitrack_a100
```

### Checkpoints
- Auto-saved to `checkpoint/` directory
- Best model tracking via validation loss
- Logs to `test_outputs/` and Weights & Biases (if configured)

---

## Evaluation

```bash
python evaluation.py --config sample
```

**Config files** (in `config/`):
- `test_config.py` → Test dataset size, output paths
- `eval_config.py` → Evaluation metrics, beam search parameters

**Output:**
- MIDI files: `eval_outputs/{maestro,slakh_stems}/sample_*.mid`
- Metrics JSON: `eval_outputs/metrics_sample.json` (timing, note accuracy, etc.)

---

## Audio Processing Pipeline

```
WAV Audio (16 kHz)
    ↓
[librosa.load @ 16kHz]
    ↓
Waveform (mono, float32)
    ↓
[spectrogram.py: STFT + Mel filter]
    ↓
Mel-Spectrogram (512 bins, 128-7600 Hz)
    ↓
[Normalize & pad to 512 time steps]
    ↓
T5 Encoder (6 layers)
    ↓
Hidden representations (encoder output)
    ↓
T5 Decoder (6 layers, autoregressive)
    ↓
Token logits (vocab_size=1536)
    ↓
[argmax → Token indices]
    ↓
[Vocabulary decode: token → event]
    ↓
NoteSequence (note_on/note_off/time_shift)
    ↓
[pretty_midi write]
    ↓
MIDI File
```

**Audio parameters** (configured in `config/data_config.py`):
- Sample rate: 16 kHz
- FFT size: 2048, hop: 128
- Mel bins: 512
- Frequency range: 20–7600 Hz

---

## Project Structure

```
MT3-pytorch/
├── config/                      # Configurations
│   ├── T5config.py             # Model architecture
│   ├── data_config.py          # Dataset + audio settings
│   ├── training_config.py      # Hyperparameters
│   ├── test_config.py          # Test/eval settings
│   └── eval_config.py
│
├── model/                       # PyTorch modules
│   ├── T5.py                   # Main transformer
│   ├── Encoder.py              # Audio → hidden
│   ├── Decoder.py              # Hidden → tokens
│   ├── Attention.py            # Multi-head attention
│   ├── Layers.py               # LayerNorm, MLP, embeddings
│   └── Mask.py                 # Masking utilities
│
├── data/                        # Data pipeline
│   ├── spectrogram.py          # STFT + Mel transform
│   ├── maestro_loader.py       # MAESTRO dataset
│   ├── slakh_loader.py         # SLAKH dataset variants
│   ├── multitask_dataset.py    # Combined dataset
│   ├── vocabularies.py         # Token ↔ MIDI codec
│   ├── training_utils.py       # Batch collation, augmentation
│   └── constants.py            # Vocab size, pad tokens
│
├── training/
│   └── train_multitrack.py      # Main training script
│
├── tests/                       # 10 test modules
│   ├── test_model_architecture.py
│   ├── test_training.py
│   ├── test_vocabularies.py
│   └── ...
│
├── pyproject.toml               # Dependencies & metadata
├── train.py                     # Legacy training (piano-only)
└── evaluation.py                # Inference & metrics
```
---

## Model Architecture

**T5 Transformer** (Encoder-Decoder):
- **Embedding**: 512 dimensions
- **Heads**: 8 (head_dim=64)
- **Layers**: 6 encoder + 6 decoder
- **Feed-forward**: 2048 intermediate dims
- **Vocab**: 1536 tokens (instruments, pitches, timing, special)
- **Dropout**: 0.1
- **Activation**: ReLU

**Inference**: Autoregressive decoding with beam search (configurable in `eval_config.py`)

---
