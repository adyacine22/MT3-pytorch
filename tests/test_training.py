"""
Priority 2: Training Process Tests
Tests the training pipeline with real data on CPU, MPS, and CUDA.
"""

import unittest
import sys
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import librosa
import note_seq
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.test_config import TEST_CONFIG, get_maestro_sample_files
from config.T5config import Magenta_T5Config
from model.T5 import Transformer
from data.spectrogram import MelSpectrogram
from data import vocabularies

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(TEST_CONFIG["output_dir"] / "test_training.log"),
    ],
)
logger = logging.getLogger(__name__)


def get_device():
    """Get available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class TestTrainingProcess(unittest.TestCase):
    """Test training pipeline components."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PROCESS TESTS - Priority 2")
        logger.info("=" * 80)

        cls.device = get_device()
        logger.info(f"Testing on device: {cls.device}")

        cls.config = Magenta_T5Config()
        cls.model = Transformer(config=cls.config)
        cls.model = cls.model.to(cls.device)

        cls.mel_extractor = MelSpectrogram(
            n_mels=TEST_CONFIG["num_mel_bins"],
            sample_rate=TEST_CONFIG["sample_rate"],
            filter_length=TEST_CONFIG["fft_size"],
            hop_length=TEST_CONFIG["hop_width"],
            win_length=TEST_CONFIG["fft_size"],
        ).to(cls.device)

        cls.vocab = vocabularies.build_codec(vocab_config=vocabularies.VocabularyConfig())

        logger.info(f"Model config: {cls.config.__class__.__name__}")
        logger.info(f"Vocabulary size: {cls.config.vocab_size}")
        logger.info(f"Device: {cls.device}")

    def test_loss_computation(self):
        """Test loss computation with cross-entropy."""
        logger.info("\n--- Test 2.1: Loss Computation ---")

        batch_size = 2
        seq_len = 50
        vocab_size = self.config.vocab_size

        # Create dummy predictions and targets
        logits = torch.randn(batch_size, seq_len, vocab_size).to(self.device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)

        logger.info(f"  Logits shape: {logits.shape}")
        logger.info(f"  Targets shape: {targets.shape}")

        # Compute loss
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

        logger.info(f"  Loss value: {loss.item():.6f}")

        # Validate loss
        self.assertFalse(torch.isnan(loss), "Loss should not be NaN")
        self.assertFalse(torch.isinf(loss), "Loss should not be Inf")
        self.assertGreater(loss.item(), 0, "Loss should be positive")

        logger.info("  ✓ Loss computation successful")

    def test_optimizer_step(self):
        """Test optimizer updates model parameters."""
        logger.info("\n--- Test 2.2: Optimizer Step ---")

        self.model.train()

        # Create optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)

        # Store initial parameters
        initial_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()

        logger.info(f"  Tracking {len(initial_params)} parameters")

        # Create dummy batch
        batch_size = 2
        encoder_seq_len = 128
        decoder_seq_len = 50

        encoder_input = torch.randn(batch_size, encoder_seq_len, self.config.emb_dim).to(
            self.device
        )
        decoder_input = torch.randint(1, self.config.vocab_size, (batch_size, decoder_seq_len)).to(
            self.device
        )
        decoder_target = torch.randint(1, self.config.vocab_size, (batch_size, decoder_seq_len)).to(
            self.device
        )

        # Forward pass
        outputs = self.model(encoder_input, decoder_target, decoder_input_tokens=decoder_input)

        # Compute loss
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(outputs.reshape(-1, self.config.vocab_size), decoder_target.reshape(-1))

        logger.info(f"  Initial loss: {loss.item():.6f}")

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check parameters changed
        changed_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in initial_params:
                if not torch.equal(param.data, initial_params[name]):
                    changed_params += 1

        logger.info(f"  Parameters updated: {changed_params}/{len(initial_params)}")

        self.assertGreater(changed_params, 0, "At least some parameters should be updated")

        self.model.eval()
        logger.info("  ✓ Optimizer step successful")

    def test_training_step_with_real_data(self):
        """Test complete training step with real MAESTRO data."""
        logger.info("\n--- Test 2.3: Training Step with Real Data ---")

        maestro_samples = get_maestro_sample_files(split="train", num_samples=1)
        if not maestro_samples:
            self.skipTest("No MAESTRO samples available")

        sample = maestro_samples[0]
        logger.info(f"  Audio file: {Path(sample['audio']).name}")
        logger.info(f"  MIDI file: {Path(sample['midi']).name}")

        # Load audio (2.048 seconds)
        audio, sr = librosa.load(sample["audio"], sr=TEST_CONFIG["sample_rate"], duration=2.048)

        # Convert to mel spectrogram
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        audio_tensor = audio_tensor / (torch.abs(audio_tensor).max() + 1e-8)

        with torch.no_grad():
            mel = self.mel_extractor(audio_tensor)
            mel = mel.transpose(1, 2)  # [1, time, n_mels]

        logger.info(f"  Mel shape: {mel.shape}")

        # Load and encode MIDI
        note_sequence = note_seq.midi_file_to_note_sequence(sample["midi"])
        note_sequence = note_seq.apply_sustain_control_changes(note_sequence)

        # Trim to 2.048 seconds - use note_seq function
        note_sequence = note_seq.extract_subsequence(note_sequence, 0.0, 2.048)

        # Encode to tokens using utils
        from data import utils

        events = utils.note_sequence_to_timed_events(note_sequence)

        # Create frame times (100 fps)
        frame_times = np.arange(0, 2.048, 1.0 / 100.0)

        # Convert events to tokens
        tokens_list, _, _ = utils.timed_events_to_tokens(events, self.vocab, frame_times)
        tokens = tokens_list if len(tokens_list) > 0 else [0]

        # Pad/truncate to fixed length
        max_len = 256
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        decoder_input = torch.LongTensor([tokens]).to(self.device)
        decoder_target = decoder_input.clone()

        logger.info(f"  Decoder input shape: {decoder_input.shape}")
        logger.info(f"  Non-zero tokens: {(decoder_input != 0).sum().item()}")

        # Training step
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Forward
        outputs = self.model(mel, decoder_target, decoder_input_tokens=decoder_input)

        # Loss
        loss = criterion(outputs.reshape(-1, self.config.vocab_size), decoder_target.reshape(-1))

        logger.info(f"  Loss: {loss.item():.6f}")

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        has_grad = sum(1 for p in self.model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in self.model.parameters())
        logger.info(f"  Gradients computed: {has_grad}/{total_params}")

        # Optimizer step
        optimizer.step()

        # Validate
        self.assertFalse(torch.isnan(loss), "Loss should not be NaN")
        self.assertGreater(has_grad, 0, "Should have gradients")

        self.model.eval()
        logger.info("  ✓ Training step with real data successful")

    def test_batch_training(self):
        """Test training with batched data."""
        logger.info("\n--- Test 2.4: Batch Training ---")

        batch_size = 4
        encoder_seq_len = 128
        decoder_seq_len = 100

        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Create batch
        encoder_input = torch.randn(batch_size, encoder_seq_len, self.config.emb_dim).to(
            self.device
        )
        decoder_input = torch.randint(1, self.config.vocab_size, (batch_size, decoder_seq_len)).to(
            self.device
        )
        decoder_target = torch.randint(1, self.config.vocab_size, (batch_size, decoder_seq_len)).to(
            self.device
        )

        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Encoder input: {encoder_input.shape}")
        logger.info(f"  Decoder input: {decoder_input.shape}")

        # Forward
        outputs = self.model(encoder_input, decoder_target, decoder_input_tokens=decoder_input)

        # Loss
        loss = criterion(outputs.reshape(-1, self.config.vocab_size), decoder_target.reshape(-1))

        logger.info(f"  Loss: {loss.item():.6f}")

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        max_grad_norm = 1.0
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        logger.info(f"  Gradient norm: {grad_norm:.6f}")
        logger.info(f"  Clipped to: {max_grad_norm}")

        # Optimizer step
        optimizer.step()

        # Validate
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(grad_norm))

        self.model.eval()
        logger.info("  ✓ Batch training successful")

    def test_gradient_accumulation(self):
        """Test gradient accumulation for larger effective batch size."""
        logger.info("\n--- Test 2.5: Gradient Accumulation ---")

        accumulation_steps = 4
        batch_size = 2
        encoder_seq_len = 128
        decoder_seq_len = 50

        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        logger.info(f"  Accumulation steps: {accumulation_steps}")
        logger.info(f"  Batch size per step: {batch_size}")
        logger.info(f"  Effective batch size: {batch_size * accumulation_steps}")

        optimizer.zero_grad()
        total_loss = 0

        for step in range(accumulation_steps):
            # Create batch
            encoder_input = torch.randn(batch_size, encoder_seq_len, self.config.emb_dim).to(
                self.device
            )
            decoder_input = torch.randint(
                1, self.config.vocab_size, (batch_size, decoder_seq_len)
            ).to(self.device)
            decoder_target = torch.randint(
                1, self.config.vocab_size, (batch_size, decoder_seq_len)
            ).to(self.device)

            # Forward
            outputs = self.model(encoder_input, decoder_target, decoder_input_tokens=decoder_input)

            # Loss
            loss = criterion(
                outputs.reshape(-1, self.config.vocab_size), decoder_target.reshape(-1)
            )
            loss = loss / accumulation_steps  # Scale loss

            # Backward
            loss.backward()

            total_loss += loss.item()
            logger.info(f"    Step {step + 1}/{accumulation_steps}: loss = {loss.item():.6f}")

        # Optimizer step after accumulation
        optimizer.step()

        logger.info(f"  Total accumulated loss: {total_loss:.6f}")

        self.assertFalse(torch.isnan(torch.tensor(total_loss)))
        self.model.eval()
        logger.info("  ✓ Gradient accumulation successful")

    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling during training."""
        logger.info("\n--- Test 2.6: Learning Rate Scheduling ---")

        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)

        # Create scheduler (warmup then decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

        initial_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"  Initial LR: {initial_lr:.6f}")

        lrs = [initial_lr]
        for epoch in range(10):
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)
            logger.info(f"    Epoch {epoch + 1}: LR = {current_lr:.6f}")

        final_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"  Final LR: {final_lr:.6f}")

        # Validate LR changed
        self.assertNotEqual(initial_lr, final_lr, "LR should change with scheduling")
        self.assertLess(final_lr, initial_lr, "LR should decrease with cosine annealing")

        logger.info("  ✓ Learning rate scheduling successful")

    def test_checkpoint_save_load(self):
        """Test saving and loading model checkpoints."""
        logger.info("\n--- Test 2.7: Checkpoint Save/Load ---")

        checkpoint_path = TEST_CONFIG["output_dir"] / "test_checkpoint.pth"

        # Train for one step to change parameters
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        encoder_input = torch.randn(2, 100, self.config.emb_dim).to(self.device)
        decoder_input = torch.randint(1, self.config.vocab_size, (2, 50)).to(self.device)
        decoder_target = torch.randint(1, self.config.vocab_size, (2, 50)).to(self.device)

        outputs = self.model(encoder_input, decoder_target, decoder_input_tokens=decoder_input)
        loss = criterion(outputs.reshape(-1, self.config.vocab_size), decoder_target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"  Saved checkpoint to: {checkpoint_path.name}")

        # Load checkpoint into new model
        new_model = Transformer(config=self.config).to(self.device)
        new_optimizer = optim.AdamW(new_model.parameters(), lr=1e-4)

        loaded_checkpoint = torch.load(checkpoint_path, map_location=self.device)
        new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
        new_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])

        logger.info(f"  Loaded checkpoint")
        logger.info(f"  Saved loss: {loaded_checkpoint['loss']:.6f}")

        # Verify parameters match
        new_model.eval()
        self.model.eval()

        with torch.no_grad():
            original_output = self.model(
                encoder_input, decoder_target, decoder_input_tokens=decoder_input
            )
            loaded_output = new_model(
                encoder_input, decoder_target, decoder_input_tokens=decoder_input
            )

        # Check outputs are identical
        max_diff = (original_output - loaded_output).abs().max().item()
        logger.info(f"  Max output difference: {max_diff:.10f}")

        self.assertLess(max_diff, 1e-5, "Loaded model should produce same outputs")

        # Cleanup
        checkpoint_path.unlink()
        logger.info("  ✓ Checkpoint save/load successful")

    def test_mixed_precision_compatibility(self):
        """Test model compatibility with automatic mixed precision."""
        logger.info("\n--- Test 2.8: Mixed Precision Compatibility ---")

        # Skip if not on CUDA (AMP primarily for CUDA)
        if self.device.type != "cuda":
            logger.info("  Skipping (AMP requires CUDA)")
            self.skipTest("Mixed precision requires CUDA")

        from torch.amp.autocast_mode import autocast
        from torch.amp.grad_scaler import GradScaler

        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        scaler = GradScaler(device="cuda")

        encoder_input = torch.randn(2, 100, self.config.emb_dim).to(self.device)
        decoder_input = torch.randint(1, self.config.vocab_size, (2, 50)).to(self.device)
        decoder_target = torch.randint(1, self.config.vocab_size, (2, 50)).to(self.device)

        # Forward with autocast
        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(encoder_input, decoder_target, decoder_input_tokens=decoder_input)
            loss = criterion(
                outputs.reshape(-1, self.config.vocab_size), decoder_target.reshape(-1)
            )

        logger.info(f"  Loss (FP16): {loss.item():.6f}")

        # Backward with scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        self.assertFalse(torch.isnan(loss))

        self.model.eval()
        logger.info("  ✓ Mixed precision training successful")


if __name__ == "__main__":
    logger.info("Starting Training Process Tests (Priority 2)")
    logger.info(f"Test output directory: {TEST_CONFIG['output_dir']}")
    logger.info(f"Log file: {TEST_CONFIG['output_dir']}/test_training.log")

    unittest.main(verbosity=2)
