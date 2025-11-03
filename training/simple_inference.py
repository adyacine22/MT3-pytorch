"""
Inference script for MT3-PyTorch.
Run inference on a single audio sample and generate MIDI output.
"""

import sys
from pathlib import Path
import logging
import warnings
import torch
from torch import inference_mode
import numpy as np
import librosa
import soundfile as sf
import note_seq

# Suppress warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
warnings.filterwarnings("ignore", category=UserWarning, module="pydub")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.inference_config import get_inference_config
from config.T5config import Magenta_T5Config
from model.T5 import Transformer
from data.maestro_loader import MIDIDataset
from data.slakh_loader import SLAKHStemDataset
from data import vocabularies
from data import utils as data_utils
from data.spectrogram import MelSpectrogram
from data.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_NUM_MEL_BINS,
    FFT_SIZE,
    DEFAULT_HOP_WIDTH,
    MEL_FMIN,
    MEL_FMAX,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize model
    model_config = Magenta_T5Config()
    model = Transformer(config=model_config, use_flash_attention=False)
    
    # Handle state dict from torch.compile()
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    logger.info(f"✓ Model loaded (step {checkpoint.get('step', checkpoint.get('epoch', 'unknown'))})")
    return model


def load_audio(audio_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE, max_length: float = 30.0):
    """
    Load audio file and convert to waveform.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        max_length: Maximum audio length in seconds
        
    Returns:
        numpy array of audio samples
    """
    logger.info(f"Loading audio from {audio_path}...")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Trim to max length
    max_samples = int(max_length * sample_rate)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        logger.info(f"  Trimmed to {max_length}s ({len(audio)} samples)")
    else:
        logger.info(f"  Loaded {len(audio)/sample_rate:.2f}s ({len(audio)} samples)")
    
    return audio


def audio_to_spectrogram(audio: np.ndarray, device: torch.device):
    """
    Convert audio waveform to mel spectrogram.
    
    Args:
        audio: Audio waveform (numpy array)
        device: Device to put tensor on
        
    Returns:
        Mel spectrogram tensor [1, mel_bins, time_frames]
    """
    # Create mel spectrogram converter
    mel_converter = MelSpectrogram(
        n_mels=DEFAULT_NUM_MEL_BINS,
        sample_rate=DEFAULT_SAMPLE_RATE,
        filter_length=FFT_SIZE,
        hop_length=DEFAULT_HOP_WIDTH,
        mel_fmin=MEL_FMIN,
        mel_fmax=MEL_FMAX,
    ).to(device)
    
    # Convert to tensor and add batch dimension
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
    
    # Compute mel spectrogram
    with torch.no_grad():
        mel_spec = mel_converter(audio_tensor)
    
    # Apply log scaling
    mel_spec = torch.log(mel_spec + 1e-6)
    
    logger.info(f"  Spectrogram shape: {mel_spec.shape}")
    return mel_spec


def load_sample_from_dataset(config: dict):
    """
    Load a sample from dataset if no audio path provided.
    
    Args:
        config: Inference configuration
        
    Returns:
        Tuple of (sample_dict, output_name, audio_waveform)
    """
    dataset_type = config["dataset_type"]
    split = config["dataset_split"]
    sample_idx = config["sample_index"]
    
    logger.info(f"Loading sample {sample_idx} from {dataset_type} {split} dataset...")
    
    if dataset_type == "maestro":
        dataset = MIDIDataset(split=split)
    elif dataset_type == "slakh":
        dataset = SLAKHStemDataset(split=split)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Get sample by iterating (IterableDataset)
    sample = None
    for i, s in enumerate(dataset):
        if i == sample_idx:
            sample = s
            break
        if i > sample_idx:  # Safety check
            break
    
    if sample is None:
        logger.error(f"Could not load sample {sample_idx}")
        raise ValueError(f"Sample {sample_idx} not found in dataset")
    
    # Extract audio and metadata
    logger.info(f"  Loaded sample from dataset (input shape: {sample['inputs'].shape})")
    
    # Try to load the original audio file if source_file is available
    audio_waveform = None
    if "source_file" in sample:
        source_file = sample["source_file"]
        logger.info(f"  Source MIDI file: {source_file}")
        
        # Convert MIDI path to audio path
        audio_file = source_file.replace(".midi", ".mp3").replace(".mid", ".mp3")
        if not Path(audio_file).exists():
            audio_file = source_file.replace(".midi", ".wav").replace(".mid", ".wav")
        
        if Path(audio_file).exists():
            try:
                audio_waveform, _ = librosa.load(
                    audio_file, 
                    sr=config["sample_rate"],
                    duration=config["max_audio_length"]
                )
                logger.info(f"  Loaded audio from {audio_file} ({len(audio_waveform)/config['sample_rate']:.2f}s)")
            except Exception as e:
                logger.warning(f"  Could not load audio: {e}")
    
    return sample, f"{dataset_type}_sample_{sample_idx}", audio_waveform


@inference_mode()
def run_inference(model: torch.nn.Module, inputs: torch.Tensor, config: dict):
    """
    Run inference on input spectrogram.
    
    Args:
        model: Model in eval mode
        inputs: Input spectrogram [1, mel_bins, time_frames]
        config: Inference configuration
        
    Returns:
        Predicted token sequence
    """
    logger.info("Running inference...")
    
    max_length = config["max_decode_length"]
    
    # Generate predictions
    predictions = model.generate(inputs, max_length=max_length)
    
    logger.info(f"  Generated {predictions.shape[1]} tokens")
    return predictions


def decode_tokens_to_midi(tokens: torch.Tensor, output_path: Path, codec: vocabularies.Codec):
    """
    Decode token sequence to MIDI file.
    
    Args:
        tokens: Token sequence [1, T]
        output_path: Path to save MIDI file
        codec: Vocabulary codec
    """
    logger.info("Decoding tokens to MIDI...")
    
    # Convert to list and remove batch dimension
    token_list = tokens[0].cpu().tolist()
    
    # Decode tokens manually without stopping at EOS
    # The vocab.decode() stops at first EOS, but we want all tokens
    vocab = vocabularies.GenericTokenVocabulary(codec.num_classes)
    
    # Manual decoding without EOS stopping
    decoded_tokens = []
    for token_id in token_list:
        if token_id == vocab.eos_id:
            # Skip EOS tokens but don't stop
            continue
        elif token_id < 3 or token_id >= vocab.vocab_size - vocab.extra_ids:
            # Skip special tokens and extra tokens
            continue
        else:
            # Convert vocabulary ID to event index
            decoded_tokens.append(token_id - 3)
    
    logger.info(f"  Decoded {len(decoded_tokens)} valid tokens from {len(token_list)} total tokens")
    
    if not decoded_tokens:
        logger.warning("  No valid tokens to decode!")
        return
    
    try:
        # Convert tokens to NoteSequence using the utility function
        note_sequence = data_utils.tokens_to_note_sequence(decoded_tokens, codec)
        
        # Count notes
        num_notes = len(note_sequence.notes)
        logger.info(f"  Generated NoteSequence with {num_notes} notes")
        
        if num_notes == 0:
            logger.warning("  No notes generated!")
            return
        
        # Save as MIDI
        note_seq.note_sequence_to_midi_file(note_sequence, str(output_path))
        logger.info(f"  ✓ Saved MIDI to {output_path}")
        
    except Exception as e:
        logger.error(f"  Error decoding to MIDI: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main inference function."""
    logger.info("=" * 80)
    logger.info("MT3 INFERENCE")
    logger.info("=" * 80)
    
    # Load configuration
    config = get_inference_config()
    
    # Get device
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Load model
    checkpoint_path = config["checkpoint_path"]
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Available checkpoints:")
        for ckpt in checkpoint_path.parent.glob("*.pt"):
            logger.info(f"  - {ckpt.name}")
        return
    
    model = load_model(checkpoint_path, device)
    
    # Initialize codec
    codec = vocabularies.build_codec(vocabularies.VocabularyConfig())
    
    # Variables for saving ground truth and audio
    sample = None
    audio_waveform = None
    
    # Load audio/sample
    if config["audio_path"] is not None:
        # Load from file
        audio_waveform = load_audio(
            config["audio_path"],
            config["sample_rate"],
            config["max_audio_length"]
        )
        inputs = audio_to_spectrogram(audio_waveform, device)
        output_name = Path(config["audio_path"]).stem
    else:
        # Load from dataset
        sample, output_name, audio_waveform = load_sample_from_dataset(config)
        inputs = sample["inputs"].unsqueeze(0).to(device)
    
    logger.info(f"\nInput shape: {inputs.shape}")
    
    # Run inference
    predictions = run_inference(model, inputs, config)
    
    # Save outputs
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predicted MIDI
    if config["save_midi"]:
        midi_path = output_dir / f"{output_name}_predicted.mid"
        decode_tokens_to_midi(predictions, midi_path, codec)
    
    # Save ground truth MIDI (if available from dataset)
    if sample is not None and "targets" in sample:
        logger.info("\nSaving ground truth MIDI...")
        try:
            # Decode ground truth tokens (without stopping at EOS)
            target_tokens = sample["targets"].cpu().tolist()
            vocab = vocabularies.GenericTokenVocabulary(codec.num_classes)
            
            # Manual decoding without EOS stopping
            decoded_targets = []
            for token_id in target_tokens:
                if token_id == vocab.eos_id or token_id == 0:  # Skip EOS and PAD
                    continue
                elif token_id < 3 or token_id >= vocab.vocab_size - vocab.extra_ids:
                    # Skip special tokens
                    continue
                else:
                    decoded_targets.append(token_id - 3)
            
            if decoded_targets:
                gt_note_sequence = data_utils.tokens_to_note_sequence(decoded_targets, codec)
                gt_midi_path = output_dir / f"{output_name}_ground_truth.mid"
                note_seq.note_sequence_to_midi_file(gt_note_sequence, str(gt_midi_path))
                logger.info(f"  ✓ Saved ground truth MIDI to {gt_midi_path}")
                logger.info(f"  Ground truth has {len(gt_note_sequence.notes)} notes")
        except Exception as e:
            logger.warning(f"  Could not save ground truth MIDI: {e}")
            import traceback
            traceback.print_exc()
    
    # Save audio input (if available)
    if audio_waveform is not None:
        audio_path = output_dir / f"{output_name}_input.wav"
        sf.write(audio_path, audio_waveform, config["sample_rate"])
        logger.info(f"\n✓ Saved input audio to {audio_path}")
    elif sample is not None and hasattr(sample, 'get') and 'audio' in sample:
        # Try to save audio from dataset sample if available
        try:
            audio_path = output_dir / f"{output_name}_input.wav"
            sample_audio = sample['audio']
            if isinstance(sample_audio, torch.Tensor):
                sample_audio = sample_audio.cpu().numpy()
            sf.write(audio_path, sample_audio, config["sample_rate"])
            logger.info(f"\n✓ Saved input audio to {audio_path}")
        except Exception as e:
            logger.warning(f"\n  Could not save input audio: {e}")
    
    if config["save_tokens"]:
        tokens_path = output_dir / f"{output_name}_predicted_tokens.txt"
        with open(tokens_path, 'w') as f:
            token_list = predictions[0].cpu().tolist()
            f.write('\n'.join(map(str, token_list)))
        logger.info(f"✓ Saved predicted tokens to {tokens_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info(f"\nGenerated files:")
    logger.info(f"  - {output_name}_predicted.mid (model prediction)")
    if sample is not None and "targets" in sample:
        logger.info(f"  - {output_name}_ground_truth.mid (reference)")
    logger.info(f"  - {output_name}_predicted_tokens.txt (token sequence)")


if __name__ == "__main__":
    main()
