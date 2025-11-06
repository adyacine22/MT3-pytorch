
# Title : Audio-to-MIDI Transcription with Instrument Embedding Conditioning and Multitrack Sequence Decoding

# Datasets details and structure : 

## 1. MAESTRO (MIDI + Audio for Solo Piano)

### Raw structure (as distributed)

- Each track: one audio file WAV + one aligned MIDI file. 
    
- Metadata file (CSV/JSON) listing `midi_filename`, `audio_filename`, `duration`, `split`, `composer`, `title`, `year`. 
    
- Proposed splits: train / validation / test.
    
- Example folder tree:
    
    ```
    maestro-v3.0.0/
	    2004/ 
	    2005/ 
	    ...
	    maestro-v3.0.0.json
	    maestro-v3.0.0.csv
      
    ```
    

### Known size / sample counts

- Duration: ~200 hours of paired audio + MIDI. 
    
- Split information: Usually the original dataset defines splits so same composition doesn’t appear across splits. 
    

## 2. Slakh2100 (Synthesized multi-track mixes + stems + MIDI)

### Raw structure

    
    ```
    Slakh2100/
	Train 
		Track00001/
	        stems/ 
		        stem001.flac
		        stem002.flac
	        midi/
		        track001.mid
		        track002.mid
	        full_mix.mid
	        full_mix.flac
	      Track00002/
	      …
	Validate
	Test
      
    ```
    
### Known size / sample counts

- Number of tracks: 2,100 (for the Slakh2100 release).
    
- Duration: ~145 hours of mixtures. 
- Download size: ~105 GB for the compressed release;

### Summary table

| Dataset   | Tracks                  | Duration   | Notes                                                              |
| --------- | ----------------------- | ---------- | ------------------------------------------------------------------ |
| Slakh2100 | 2,100 tracks (full mix) | ~145 hours | Multi-instrument, stems + MIDI, variable number of stems per track |

# Step 1 : Vanilla MT3 in PyTorch

## 1.  Unified Dataset + Index Indexing Layer


```python
{
  "track_id": str,
  "dataset": Literal["maestro", "slakh","slakh_stems"],
  "audio_path": str,                     # mono or stereo wav/flac; will be resampled to 16k
  "midi_path": str,                      # .mid or .midi; 
  "instrument_programs": List[int],            # [0..127], drums as 128, Note : 0 for maestro 
  "instrument_names": List[str], # Note : Acoustic Piano for maestro             
  "instrument_classes": List[str], # Note : Piano  for maestro
  "split": Literal["train", "validate ","test"]              
}
```

---

## 2. Audio Pipeline (Offline preprocessing + On-the-fly GPU features)


### **Offline CPU preprocessing (run once before training)** : 

- Load WAV/FLAC, mono resample to 16 kHz
    
- DC removal → HP filter → loudness normalize → optional pre-emphasis → dither
    
- Chunk into fixed length: **32768 samples = 2.048 s**
    
- Store as: `float16` or `float32` tensor `[C, 32768]` per chunk
    

### **On-the-fly GPU pipeline at training**: 

- Random chunk selection per track with temperature based selection from datasets
    
- Waveform augmentation : 
    
- STFT → magnitude → Mel projection → log compression
     

```json
{
  "audio_pipeline": [
    {
      "id": 1,
      "name": "Load audio",
      "stage": "offline",
      "split": "all",
      "computation": "cpu",
      "description": "Load waveform, convert to mono, resample to 16 kHz",
      "input_shape": "[D, N, C]",
      "output_shape": "[D, N', 1]",
      "params": {
        "target_sample_rate": 16000,
        "convert_to_mono": true,
        "dtype": "float32",
        "normalize_db": null,
        "backend": "librosa"
      }
    },
    {
      "id": 2,
      "name": "DC removal + High-pass filter",
      "stage": "offline",
      "split": "all",
      "computation": "cpu",
      "description": "Remove DC bias and rumble below cutoff",
      "input_shape": "[D, N', 1]",
      "output_shape": "[D, N', 1]",
      "params": {
        "dc_block": true,
        "hp_cutoff_hz": 30.0,
        "filter_order": 2,
        "filter_type": "biquad",
        "zero_phase": true
      }
    },
    {
      "id": 3,
      "name": "Loudness normalization",
      "stage": "offline",
      "split": "all",
      "computation": "cpu",
      "description": "Normalize perceptual loudness to target LUFS",
      "input_shape": "[D, N', 1]",
      "output_shape": "[D, N', 1]",
      "params": {
        "target_lufs": -23.0,
        "loudness_standard": "ITU-R BS.1770-4",
        "gain_clamp_db": 12.0,
        "true_peak_limit_dbfs": -1.0,
        "integrated_window_sec": 3.0
      }
    },
    {
      "id": 4,
      "name": "Pre-emphasis filter",
      "stage": "offline",
      "split": "all",
      "computation": "cpu",
      "description": "Boost high-frequency energy",
      "input_shape": "[D, N', 1]",
      "output_shape": "[D, N', 1]",
      "params": {
        "alpha": 0.97,
        "apply": true
      }
    },
    {
      "id": 5,
      "name": "Dither",
      "stage": "offline",
      "split": "all",
      "computation": "cpu",
      "description": "Add low-level TPDF dither before chunking",
      "input_shape": "[D, N', 1]",
      "output_shape": "[D, N', 1]",
      "params": {
        "type": "TPDF",
        "amplitude": 1e-5,
        "enabled": true
      }
    },
    {
      "id": 6,
      "name": "Offline chunking",
      "stage": "offline",
      "split": "all",
      "computation": "cpu",
      "description": "Slice waveform into fixed-length chunks",
      "input_shape": "[D, N', 1]",
      "output_shape": "[D, NumChunks, 32768]",
      "params": {
        "chunk_samples": 32768,
        "hop_samples": 32768,
        "pad_final": true,
        "pad_mode": "constant",
        "pad_value": 0.0,
        "drop_last": false,
        "shuffle_index": false
      }
    },
    {
      "id": 7,
      "name": "Random chunk sampling",
      "stage": "on-the-fly",
      "split": "train",
      "computation": "cpu",
      "description": "Sample one chunk per track per epoch",
      "input_shape": "[D, NumChunks, 32768]",
      "output_shape": "[B, 32768]",
      "params": {
        "chunks_per_epoch": 1,
        "sampling_strategy": "uniform",
        "temperature": 1.0,
        "seed_base": 1337
      }
    },
    {
      "id": 8,
      "name": "Waveform augmentation",
      "stage": "on-the-fly",
      "split": "train",
      "computation": "cpu",
      "description": "Apply RIR, noise, EQ, and light distortions",
      "input_shape": "[B, 32768]",
      "output_shape": "[B, 32768]",
      "params": {
        "rir_prob": 0.3,
        "rir_rt60_max": 1.0,
        "noise_prob": 0.5,
        "snr_db_min": 10.0,
        "snr_db_max": 30.0,
        "eq_prob": 0.3,
        "eq_gain_db_max": 6.0,
        "clip_prob": 0.1,
        "clip_threshold_dbfs": -3.0,
        "pitch_shift_prob": 0.0,
        "time_stretch_prob": 0.0
      }
    },
    {
      "id": 9,
      "name": "STFT",
      "stage": "on-the-fly",
      "split": "all",
      "computation": "gpu",
      "description": "Waveform to complex spectrogram",
      "input_shape": "[B, 32768]",
      "output_shape": "[B, 256, 1025]",
      "params": {
        "n_fft": 2048,
        "win_length": 2048,
        "hop_length": 128,
        "window": "hann",
        "center": true,
        "pad_end": true,
        "return_complex": true
      }
    },
    {
      "id": 10,
      "name": "Magnitude → Mel projection",
      "stage": "on-the-fly",
      "split": "all",
      "computation": "gpu",
      "description": "Project magnitude spectrogram to Mel scale",
      "input_shape": "[B, 256, 1025]",
      "output_shape": "[B, 256, 512]",
      "params": {
        "power": 1.0,
        "n_mels": 512,
        "fmin_hz": 20.0,
        "fmax_hz": 8000.0,
        "mel_norm": "slaney"
      }
    },
    {
      "id": 11,
      "name": "Log compression",
      "stage": "on-the-fly",
      "split": "all",
      "computation": "gpu",
      "description": "Stabilize dynamic range with log transform",
      "input_shape": "[B, 256, 512]",
      "output_shape": "[B, 256, 512]",
      "params": {
        "type": "log",
        "epsilon": 1e-5
      }
    },
    {
      "id": 12,
      "name": "Learnable per-frequency normalization",
      "stage": "on-the-fly",
      "split": "all",
      "computation": "gpu",
      "description": "Adaptive normalization over frequency axis",
      "input_shape": "[B, 256, 512]",
      "output_shape": "[B, 256, 512]",
      "params": {
        "method": "layernorm",
        "axis": "freq",
        "affine": true,
        "eps": 1e-5
      }
    }
  ]
}


```


---

## 3. MIDI Tokenization (Option 1 Vocabulary) — Pre-Tokenized per Chunk

### **What to implement**

- Each chunk has **its own token sequence**, already aligned in time.
    
- Only **single mixed token stream** (no per-instrument splitting yet).
    
- Use **Option 1** vocabulary containing:  
    `shift`, `pitch`, `velocity`, `tie`, `program`, `drum`, `EOS`, `PAD`, `UNK`
    
- Max token length per chunk: **1024**
    
- Right padding applied + EOS enforced
    

### Option 1:  With program vocab : 

```json 
{
  "vocabulary": {
      "config": {
        "steps_per_second": 100,
        "max_shift_seconds": 10,
        "num_velocity_bins": 127
      },
      "event_ranges": [
        {"type": "pitch", "min": 21, "max": 108},
        {"type": "velocity", "min": 0, "max": 127},
        {"type": "tie", "min": 0, "max": 0},
        {"type": "program", "min": 0, "max": 127},
        {"type": "drum", "min": 21, "max": 108}
      ],
      "cumulative_vocabulary": [
        {"type": "special", "name": "PAD", "id": 0},
        {"type": "special", "name": "EOS", "id": 1},
        {"type": "special", "name": "UNK", "id": 2},
        {"type": "shift", "min_id": 3, "max_id": 1003},
        {"type": "pitch", "min_id": 1004, "max_id": 1091},
        {"type": "velocity", "min_id": 1092, "max_id": 1219},
        {"type": "tie", "id": 1220},
        {"type": "program", "min_id": 1221, "max_id": 1348},
        {"type": "drum", "min_id": 1349, "max_id": 1436}
      ],
      "total_size": 1437
    }
  }
```
###  Example : 
```json
{
  "chunk_1": {
    "description": "Piano and drums play. A sustain pedal is engaged and a piano note is held across the chunk boundary.",
    "event_format": [
      ["program", 0], ["velocity", 100], ["pitch", 60],  // t=0.0s: Piano C4 on
      ["shift", 50],
      ["velocity", 110], ["drum", 36],                   // t=0.5s: Drum Kick (no program event)
      ["shift", 50],
      ["program", 0], ["velocity", 0], ["pitch", 60],     // t=1.0s: Piano C4 off
      ["shift", 100],
      ["program", 0], ["velocity", 100], ["pitch", 64]   // t=2.0s: Piano E4 on
    ],
    "token_format": [
      1221, 1192, 1043,
      53,
      1202, 1364,  
      53,
      1221, 1092, 1043,
      103,
      1221, 1192, 1047
    ]
  },
  "chunk_2": {
    "description": "Starts with a tied piano note held by sustain, which is then released. Another drum hit follows.",
    "event_format": [
      ["tie", 0], ["program", 0], ["pitch", 64],           // t=2.25s (start of chunk): Tie Piano E4
      ["shift", 75],
      ["program", 0], ["velocity", 0], ["pitch", 64],     // t=3.0s: Piano E4 off (sustain released)
      ["shift", 50],
      ["velocity", 110], ["drum", 38]                    // t=3.5s: Drum Snare (no program event)
    ],
    "token_format": [
      1220, 1221, 1047,
      78,
      1221, 1092, 1047,
      53,
      1202, 1366 
    ]
  }
}
```



---

# **4️⃣ Dataloader + Batch Construction**

### **What to implement**

- Random **track selection** using temperature-based sampling  
- Random **chunk selection inside track** (uniform or weighted)  
- GPU STFT/mel per batch  
- Paired loading: Audio `[B, 256, 512]`, Tokens `[B, 1024]`  
- Store tensors in **channels-last** when possible  
- Final batch returned:

```
{
  "audio": FloatTensor[B, 256, 512],
  "tokens": LongTensor[B, 1024],
  "attn_mask": BoolTensor[B, 1024]
}
```

 Ignore PAD tokens in loss (`label_pad_id = -100`)

###  **JSON Config**

```json
{
  "dataloader": {
    "batch_size": 16,
    "num_workers": 8,
    "prefetch_factor": 4,
    "pin_memory": true,
    "shuffle": true,
    "sampling": {
      "track_temperature": 0.7,
      "chunk_sampling": "uniform"
    },
    "padding": {
      "token_pad_id": 0,
      "label_ignore_id": -100
    },
    "output_shapes": {
      "audio": [256, 512],
      "tokens": [1024]
    }
  }
}
```


![[Pasted image 20251105081059.png]]
## 4/ Positional Encodings

- **Encoder positions**: absolute (fixed sinusoidal or learned absolute) over **time** axis of mel frames.
    
- **Decoder positions**: absolute positional embeddings matching token positions.
    
- Keep Step 1 simple (no RoPE/relative bias yet—those come in Step 2).
    

**Config (JSON)**

```json
{
  "positions": {
    "encoder": { "type": "absolute_sinusoidal", "max_len": 4096, "dropout": 0.0 },
    "decoder": { "type": "absolute_sinusoidal", "max_len": 2048, "dropout": 0.0 }
  }
}
```

---

## 5/ Encoder (Transformer)

- **Input projection**: linear proj of mel features to `d_model`.
    
- **Stack** of `num_encoder_layers`:
    
    - Pre-LayerNorm (or Post for strict T5 behavior)
        
    - Multi-head self-attention
        
    - Feed-forward: MLP (e.g., GELU) with `mlp_dim`
        
    - Residual connections; dropout
        
- **No** hierarchical downsample/conv/strided-attn at Step 1.
    

**Config (JSON)**

```json
{
  "encoder": {
    "d_model": 512,
    "num_layers": 8,
    "num_heads": 6,
    "mlp_dim": 1024,
    "attn_dropout": 0.0,
    "resid_dropout": 0.1,
    "mlp_dropout": 0.1,
    "norm": "layernorm_pre",
    "activation": "gelu",
    "input_projection": { "type": "linear" }
  }
}
```

---

## 6/ Decoder (Transformer)

- **Embedding**: token embedding (size = vocab size) to `d_model`.
    
- **Stack** of `num_decoder_layers`:
    
    - Pre-LayerNorm
        
    - **Masked** self-attention (causal)
        
    - **Cross-attention** to encoder outputs
        
    - Feed-forward MLP
        
    - Residuals; dropout
        
- **LM head**: tied embeddings or separate linear to vocab size.
    

**Config (JSON)**

```json
{
  "decoder": {
    "d_model": 512,
    "num_layers": 8,
    "num_heads": 6,
    "mlp_dim": 1024,
    "attn_dropout": 0.0,
    "resid_dropout": 0.1,
    "mlp_dropout": 0.1,
    "norm": "layernorm_pre",
    "activation": "gelu",
    "tie_embeddings": true,
    "causal_masking": true
  },
  "lm_head": {
    "vocab_size": 1437,
    "bias": false
  }
}
```

---

## 7/ Objective, Optimization, & Scheduling (Step-1 defaults)

- **Loss**: cross-entropy on decoder labels; ignore padded labels.
    
- **Optimizer**: AdamW with standard βs and weight decay; warmup + cosine or inverse-sqrt.
    
- **Teacher forcing**: feed ground-truth tokens during training.
    
- **Regularization**: dropout only (no label smoothing yet if you want the purest baseline).
    

**Config (JSON)**

```json
{
  "optimization": {
    "optimizer": {
      "type": "adamw",
      "lr": 0.0005,
      "betas": [0.9, 0.999],
      "weight_decay": 0.01,
      "eps": 1e-8
    },
    "scheduler": {
      "type": "cosine",
      "warmup_steps": 4000,
      "min_lr_ratio": 0.1
    },
    "gradient_clipping": 1.0,
    "dropout_seed": 1234
  },
  "training": {
    "max_steps": 200000,
    "val_every_steps": 2000,
    "log_every_steps": 100
  }
}
```

---

## 8/ Inference (Segment Stitching)

- **Greedy** or small **beam search** decoding per segment until `EOS` or max length.
    
- **Concatenate** decoded segments; apply **tie logic**:
    
    - Start of segment: treat tokens before `EndTie` as already-active notes.
        
    - If a note-on in segment N lacks note-off in segment N+1, **gracefully end** at boundary (tie section protocol).
        
- (No overlap-add at Step 1; add later if needed.)
    

**Config (JSON)**

```json
{
  "inference": {
    "decoding": {
      "method": "greedy",
      "max_length": 1024,
      "eos_token_id": 1,
      "temperature": 1.0,
      "top_k": 0,
      "top_p": 1.0
    },
    "stitching": {
      "use_tie_sections": true,
      "overlap_seconds": 0.0
    }
  }
}
```

---

## 9/ Metrics & Evaluation

- **Token-level**: accuracy, loss.
    
- **Music-level** (per piece and averaged):
    
    - Onset F1, Onset+Offset F1 (with 50ms/± frames tolerance)
        
    - Frame F1 (optional), Instrument-conditioned F1 (optional at Step 1)
        
- **Logging**: per-dataset breakdown; save a few MIDI reconstructions for sanity checks.
    

**Config (JSON)**

```json
{
  "evaluation": {
    "metrics": ["token_loss", "token_accuracy", "onset_f1", "onset_offset_f1"],
    "tolerances_ms": { "onset": 50, "offset": 50 },
    "num_eval_items": 256,
    "save_predicted_midi": true,
    "samples_to_dump": 8
  }
}
```

---

## 10) Repro, Checkpointing, Runtime

- **Seeding**: Python/NumPy/PyTorch seeds and deterministic flags (as feasible).
    
- **Checkpointing**: save best-by-val-loss and periodic snapshots; include optimizer/scheduler states.
    
- **Config logging**: write the JSON config to the run folder.
    

**Config (JSON)**

```json
{
  "runtime": {
    "seed": 42,
    "deterministic": false,
    "cudnn_benchmark": true,
    "device": "cuda"
  },
  "checkpointing": {
    "save_dir": "checkpoints/step1_vanilla_mt3",
    "save_every_steps": 5000,
    "keep_last": 3,
    "keep_best": 1,
    "metric": "token_loss",
    "mode": "min"
  }
}
```

---

### Summary of Step-1 constraints

- **Vocabulary:** **Option 1** (includes program tokens)
    
- **Architecture:** plain encoder–decoder transformer (no conv/pyramid/strided/MoE yet)
    
- **Segments:** independent segments with **tie** handling
    
- **Loss:** CE on token stream; teacher forcing
    
- **Decoding:** greedy/beam per segment + stitching
    


#  STEP 2 — Modernized Training + Efficient Attention + Encoder/Decoder Improvements


## 1) Mixed Precision (AMP / BF16) (CUDA Training)

- Enable **autocast(fp16 or bf16)** for forward pass
    
- Use `GradScaler` for fp16 if needed (bf16 usually doesn’t require scaling)
    
- Ensure numerical stability in loss + softmax + logits
    

**Config**

```json
{
  "precision": {
    "dtype": "bf16", 
    "use_grad_scaler": false,
    "fallback_fp32_layers": ["lm_head", "softmax"]
  }
}
```

---

## 2) FlashAttention / SDPA

- Replace PyTorch `nn.MultiheadAttention` with **scaled-dot-product attention API**
    
- Enable FlashAttention backend automatically when GPU supports it
    
- Disable dropout inside attention if FlashAttention requires it
    

**Config**

```json
{
  "attention": {
    "backend": "flash", 
    "enable_flash": true,
    "enable_math": true,
    "enable_mem_efficient": true
  }
}
```

---

## 3) `torch.compile` (Graph Optimization)

- Wrap model with `torch.compile(model, mode="max-autotune" | "reduce-overhead")`
    
- Optional fallback to eager mode when debugging
    

**Config**

```json
{
  "compile": {
    "enabled": true,
    "mode": "max-autotune",
    "fullgraph": false
  }
}
```

---

## 4) Activation Checkpointing

- Apply checkpointing to encoder & decoder block stacks
    
- Reduce memory usage at the cost of some recomputation
    
- Keep final LM head uncheckpointed for speed
    

**Config**

```json
{
  "checkpointing": {
    "enabled": true,
    "checkpoint_encoder_blocks": true,
    "checkpoint_decoder_blocks": true,
    "granularity": "block"
  }
}
```

---

## 5) Optimizer, Scheduler, Gradient Scaling (Upgraded)

- Switch from **AdamW** to **Fused AdamW** or **PagedAdamW** if available
    
- Add **gradient clipping** + **EMA weights** for eval stability
    

**Config**

```json
{
  "optimization": {
    "optimizer": {
      "type": "fused_adamw",
      "lr": 0.0004,
      "betas": [0.9, 0.98],
      "weight_decay": 0.01,
      "eps": 1e-8
    },
    "scheduler": {
      "type": "cosine",
      "warmup_steps": 3000,
      "min_lr_ratio": 0.05
    },
    "grad_clip": 1.0,
    "ema": {
      "enabled": true,
      "decay": 0.999
    }
  }
}
```

---

## 6) Encoder/Decoder Internal Upgrades (formerly your “6.A”)

###  Switch to **RMSNorm** (pre-norm)

###  Replace FFN MLP → **SwiGLU**

###  Use **ALiBi or Relative Pos Bias or RoPE**  instead of fixed sinusoidal

**Config**

```json
{
  "model_upgrades": {
    "norm_type": "rmsnorm",
    "ffn_activation": "swiglu",
    "positional_encoding": ["alibi","relative","RoPE"]
    "residual_scale": "none"
  }
}
```

---

## 7) Training Stability & Regularization

- Label smoothing (e.g. 0.1)
    
- Dropout tuned down slightly (since RMSNorm+SwiGLU stabilizes)
    
- Dataset-balanced sampling to avoid piano-only dominance
    
- SpecAugment (light), time masking only (frequency masking optional)
    

**Config**

```json
{
  "regularization": {
    "label_smoothing": 0.1,
    "dropout": {
      "attn": 0.0,
      "residual": 0.05,
      "mlp": 0.05
    },
    "spec_augment": {
      "time_mask_width": 48,
      "freq_mask_width": 0,
      "prob": 0.5
    }
  }
}
```

---

## 8) Inference Optimizations

- KV-cache enabled for AR decoding
    
- Auto bf16 inference if device supports
    
- Optional **int8 weight-only quantization** for LM head only (zero accuracy loss)
    
- Micro-batch beam decoding
    

**Config**

```json
{
  "inference": {
    "use_kv_cache": true,
    "dtype": "bf16",
  }
}
```

---

## 9) Training Loop Differences 

| Feature             | Step 1      | Step 2                |
| ------------------- | ----------- | --------------------- |
| Precision           | FP32 only   | BF16 or FP16 AMP      |
| Attention           | Vanilla MHA | FlashAttention / SDPA |
| Norm type           | LayerNorm   | RMSNorm               |
| MLP                 | GELU        | SwiGLU                |
| Dropout             | 0.1         | 0.05 (more stable)    |
| Positional encoding | Sinusoidal  | ALiBi or RoPE         |
| Optimizer           | AdamW       | FusedAdamW + EMA      |
| Checkpointing       | Off         | On                    |
| `torch.compile`     | Off         | On                    |
| Inference cache     | Off         | KV-cache enabled      |


# Step 3 — Multimodal Instrument Conditioning 


Multimodal conditioning provides the model with explicit information about which instruments are present in the audio, reducing ambiguity in the transcription task and allowing the network to focus on _what_ each instrument plays instead of first having to infer _who_ is playing.
## 3.1 Late Conditioning at the Decoder Input (Early Fusion)

### Purpose

Give the decoder a global, constant “hint” about which instruments are present in the current segment, with minimal changes. This often improves instrument-specific token emission (e.g., fewer spurious `program` switches), especially in mixtures with a few dominant instruments.

### Data flow 

```
Instruments (program IDs) ──> Embedding ──> Pool (mean/attn)
                                        │
Audio ──> Encoder ─────────────┐        │
                               ▼        ▼
                         Decoder Input Embeddings (+ INST_CTX)
                                        │
                                        ▼
                [Dec Self-Attn] → [Dec Cross-Attn to Enc] → [FFN] → LM Head
```


* Build an **instrument pool embedding** once per sample from the set of active instruments (program IDs, optionally instrument classes).
* Inject this pooled embedding into the **decoder input**:
  **add/concat** the pooled vector to every decoder token embedding (feature-wise addition or small FiLM on embeddings).
* Encoder remains unchanged; cross-attention remains unchanged.
* Loss, vocabulary, and training loop remain unchanged.

### Parameterization


* Instrument embedding table: size ≈ 129 (0–127 programs + drums) × d_inst
* Pooling: mean or attention pooling (small MLP for attention scores)
* Optional per-token conditioning MLP if you add/concat instead of prepending

---

## 3.2 Encoder-Side Feature-wise Linear Modulation Conditioning (Early Fusion)

### Purpose

Bias the **audio representation** toward the currently present instruments, making the encoder separate timbres and harmonic cues better before the decoder sees them without adding attention blocks

### What changes

* Compute pooled instrument embedding as in 3.1.
* For each encoder block (or group of blocks), generate FiLM parameters `(γ, β)` from the pooled embedding via a small MLP.
* Apply FiLM to encoder hidden states before attention and/or before the FFN:

  * `h ← γ ⊙ h + β` (element-wise), with broadcasting over time steps.

### Data flow (text diagram)

```
Instruments ──> Embedding ──> Pool ──> MLP ──> {γ_l, β_l} 
                                         
Audio ──> Input Proj ─> Enc Block 1:  h ← γ₁ ⊙ h + β₁ → Self-Attn → FFN -> 

					 ─> Enc Block 2:  h ← γ₂ ⊙ h + β₂ → … ->
					 …
					 ─> Enc Output
					 
						  │
						  ▼
						  
				   Decoder (as in Step 1)
```

### Parameterization

* One small MLP per FiLM site (shared across layers or per-layer)
* `(γ, β)` broadcast across the time axis; optionally layer-specific dimension projections



---

## 3.3 Dual-Stream Interleaved Audio↔Instrument Attention 

### Purpose

Create two interacting streams—**Audio stream** and **Instrument stream**—and let them exchange information **layer-by-layer**, so that the audio representation becomes instrument-aware and the instrument representation becomes audio-grounded.

### What changes

* Add a lightweight **Instrument Encoder** that takes the active instrument tokens as a short sequence (length K).
* At each layer, interleave **cross-attention both ways**:
  * Update Audio using Instrument as keys/values;
  * Update Instrument using Audio as keys/values;
  * Then apply each stream’s self-attention and FFN.
* Feed the **fused Audio stream** to the decoder as usual; optionally also expose the Instrument stream to the decoder via an extra cross-attention (like 3.3), but typically the fused Audio is enough.

### Data flow (text diagram)

```
Audio ──> Input Proj ──> Audio Stream L1 … Lm
                                  ▲       │
                                  │       ▼
Instruments ──> Inst Embeds ─> Inst Stream L1 … Lm

Per layer l:
   Audio_l  ←  CrossAttn(Audio←Inst_l-1)  → Self-Attn → FFN → Audio_l
   Inst_l   ←  CrossAttn(Inst←Audio_l-1)  → Self-Attn → FFN → Inst_l

After m layers:
   Fused Audio_l → Decoder Cross-Attn → LM Head
```

### Parameterization

* Instrument stream depth: 2–4 layers (often much shallower than audio stream)
* Cross-attentions are lightweight; share projections across layers if needed
* Optional gates per direction to modulate cross-stream exchange