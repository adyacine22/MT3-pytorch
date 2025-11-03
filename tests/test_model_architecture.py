"""
Priority 1.3: Model Architecture and Forward Pass Tests
Tests the T5 Transformer model with real data on CPU, MPS, and CUDA.
"""

import unittest
import sys
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn
import librosa

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.test_config import TEST_CONFIG, get_maestro_sample_files
from config.T5config import Magenta_T5Config
from model.T5 import Transformer
from model.Encoder import Encoder
from model.Decoder import Decoder
from data.spectrogram import MelSpectrogram
from data import vocabularies

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(TEST_CONFIG["output_dir"] / "test_model_architecture.log"),
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


class TestModelArchitecture(unittest.TestCase):
    """Test model architecture and forward pass."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        logger.info("\n" + "=" * 80)
        logger.info("MODEL ARCHITECTURE TESTS - Priority 1.3")
        logger.info("=" * 80)

        cls.device = get_device()
        logger.info(f"Testing on device: {cls.device}")

        cls.config = Magenta_T5Config()
        cls.model = Transformer(config=cls.config)
        cls.model = cls.model.to(cls.device)
        cls.model.eval()

        cls.mel_extractor = MelSpectrogram(
            n_mels=TEST_CONFIG["num_mel_bins"],
            sample_rate=TEST_CONFIG["sample_rate"],
            filter_length=TEST_CONFIG["fft_size"],
            hop_length=TEST_CONFIG["hop_width"],
            win_length=TEST_CONFIG["fft_size"],
        ).to(cls.device)

        logger.info(f"Model config: {cls.config.__class__.__name__}")
        logger.info(f"Embedding dim: {cls.config.emb_dim}")
        logger.info(f"Encoder layers: {cls.config.num_encoder_layers}")
        logger.info(f"Decoder layers: {cls.config.num_decoder_layers}")
        logger.info(f"Vocabulary size: {cls.config.vocab_size}")

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        logger.info("\n--- Test 1.3.1: Model Initialization ---")

        # Check model components exist
        self.assertIsNotNone(self.model.encoder, "Model should have encoder")
        self.assertIsNotNone(self.model.decoder, "Model should have decoder")

        logger.info(f"  Model has encoder: {self.model.encoder is not None}")
        logger.info(f"  Model has decoder: {self.model.decoder is not None}")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

        self.assertGreater(total_params, 0, "Model should have parameters")
        self.assertEqual(total_params, trainable_params, "All params should be trainable")

        logger.info("  ✓ Model initialization successful")

    def test_encoder_forward(self):
        """Test encoder forward pass."""
        logger.info("\n--- Test 1.3.2: Encoder Forward Pass ---")

        batch_size = 2
        seq_len = 256
        emb_dim = self.config.emb_dim

        # Create dummy mel-spectrogram input
        mel_input = torch.randn(batch_size, seq_len, emb_dim).to(self.device)
        logger.info(f"  Input shape: {mel_input.shape}")

        # Forward through encoder
        with torch.no_grad():
            encoder_output = self.model.encoder(mel_input, deterministic=True)

        logger.info(f"  Encoder output shape: {encoder_output.shape}")
        logger.info(f"  Expected: [batch={batch_size}, seq={seq_len}, dim={emb_dim}]")

        # Validate output shape
        self.assertEqual(encoder_output.shape[0], batch_size)
        self.assertEqual(encoder_output.shape[1], seq_len)
        self.assertEqual(encoder_output.shape[2], emb_dim)

        # Check for NaN or Inf
        self.assertFalse(torch.isnan(encoder_output).any(), "Output should not contain NaN")
        self.assertTrue(torch.isfinite(encoder_output).all(), "Output should be finite")

        logger.info("  ✓ Encoder forward pass successful")

    def test_decoder_forward(self):
        """Test decoder forward pass."""
        logger.info("\n--- Test 1.3.3: Decoder Forward Pass ---")

        batch_size = 2
        encoder_seq_len = 256
        decoder_seq_len = 128
        emb_dim = self.config.emb_dim
        vocab_size = self.config.vocab_size

        # Create dummy encoder outputs
        encoder_output = torch.randn(batch_size, encoder_seq_len, emb_dim).to(self.device)

        # Create decoder input tokens
        decoder_input_ids = torch.randint(0, vocab_size, (batch_size, decoder_seq_len)).to(
            self.device
        )

        logger.info(f"  Encoder output shape: {encoder_output.shape}")
        logger.info(f"  Decoder input shape: {decoder_input_ids.shape}")

        # Forward through decoder
        with torch.no_grad():
            decoder_output, _ = self.model.decoder(
                encoded=encoder_output,
                decoder_input_tokens=decoder_input_ids,
                deterministic=True,
            )

        logger.info(f"  Decoder output shape: {decoder_output.shape}")
        logger.info(f"  Expected: [batch={batch_size}, seq={decoder_seq_len}, vocab={vocab_size}]")

        # Validate output shape
        self.assertEqual(decoder_output.shape[0], batch_size)
        self.assertEqual(decoder_output.shape[1], decoder_seq_len)
        self.assertEqual(decoder_output.shape[2], vocab_size)

        # Check for NaN or Inf
        self.assertFalse(torch.isnan(decoder_output).any(), "Output should not contain NaN")
        self.assertTrue(torch.isfinite(decoder_output).all(), "Output should be finite")

        logger.info("  ✓ Decoder forward pass successful")

    def test_full_model_forward(self):
        """Test complete model forward pass."""
        logger.info("\n--- Test 1.3.4: Full Model Forward Pass ---")

        batch_size = 2
        encoder_seq_len = 256
        decoder_seq_len = 100
        emb_dim = self.config.emb_dim
        vocab_size = self.config.vocab_size

        # Create inputs
        encoder_input = torch.randn(batch_size, encoder_seq_len, emb_dim).to(self.device)
        decoder_input_ids = torch.randint(0, vocab_size, (batch_size, decoder_seq_len)).to(
            self.device
        )
        decoder_target_ids = torch.randint(0, vocab_size, (batch_size, decoder_seq_len)).to(
            self.device
        )

        logger.info(f"  Encoder input shape: {encoder_input.shape}")
        logger.info(f"  Decoder input shape: {decoder_input_ids.shape}")
        logger.info(f"  Decoder target shape: {decoder_target_ids.shape}")

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                encoder_input,
                decoder_target_ids,
                decoder_input_tokens=decoder_input_ids,
            )

        logger.info(f"  Output shape: {outputs.shape}")

        # Validate
        self.assertEqual(outputs.shape[0], batch_size)
        self.assertEqual(outputs.shape[1], decoder_seq_len)
        self.assertEqual(outputs.shape[2], vocab_size)

        # Check for valid values
        self.assertFalse(torch.isnan(outputs).any())
        self.assertTrue(torch.isfinite(outputs).all())

        logger.info("  ✓ Full model forward pass successful")

    def test_encoder_with_real_spectrogram(self):
        """Test encoder with real audio spectrogram."""
        logger.info("\n--- Test 1.3.5: Encoder with Real Spectrogram ---")

        maestro_samples = get_maestro_sample_files(split="train", num_samples=1)
        if not maestro_samples:
            self.skipTest("No MAESTRO samples available")

        audio_path = maestro_samples[0]["audio"]
        logger.info(f"  Audio file: {Path(audio_path).name}")

        # Load audio (2.048 seconds)
        audio, sr = librosa.load(audio_path, sr=TEST_CONFIG["sample_rate"], duration=2.048)

        # Convert to mel spectrogram
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        audio_tensor = audio_tensor / (torch.abs(audio_tensor).max() + 1e-8)

        with torch.no_grad():
            mel = self.mel_extractor(audio_tensor)  # [1, n_mels, time]
            mel = mel.transpose(1, 2)  # [1, time, n_mels]

        logger.info(f"  Audio shape: {audio.shape}")
        logger.info(f"  Mel shape: {mel.shape}")
        logger.info(f"  Mel range: [{mel.min():.2f}, {mel.max():.2f}]")

        # Forward through encoder
        with torch.no_grad():
            encoder_output = self.model.encoder(mel, deterministic=True)

        logger.info(f"  Encoder output shape: {encoder_output.shape}")

        # Validate
        self.assertEqual(encoder_output.shape[0], 1)
        self.assertEqual(encoder_output.shape[2], self.config.emb_dim)
        self.assertFalse(torch.isnan(encoder_output).any())
        self.assertTrue(torch.isfinite(encoder_output).all())

        logger.info("  ✓ Encoder with real spectrogram successful")

    def test_batch_processing(self):
        """Test model with different batch sizes."""
        logger.info("\n--- Test 1.3.6: Batch Processing ---")

        batch_sizes = [1, 2, 4]
        seq_len = 128
        decoder_seq_len = 50

        for batch_size in batch_sizes:
            logger.info(f"  Testing batch size: {batch_size}")

            # Create inputs
            encoder_input = torch.randn(batch_size, seq_len, self.config.emb_dim).to(self.device)
            decoder_input = torch.randint(
                0, self.config.vocab_size, (batch_size, decoder_seq_len)
            ).to(self.device)
            decoder_target = torch.randint(
                0, self.config.vocab_size, (batch_size, decoder_seq_len)
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    encoder_input, decoder_target, decoder_input_tokens=decoder_input
                )

            # Validate
            self.assertEqual(outputs.shape[0], batch_size)
            self.assertFalse(torch.isnan(outputs).any())

            logger.info(f"    Output shape: {outputs.shape} ✓")

        logger.info("  ✓ Batch processing successful")

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        logger.info("\n--- Test 1.3.7: Gradient Flow ---")

        # Put model in train mode
        self.model.train()

        batch_size = 2
        seq_len = 128
        decoder_seq_len = 50

        # Create inputs
        encoder_input = torch.randn(batch_size, seq_len, self.config.emb_dim).to(self.device)
        encoder_input.requires_grad = True

        decoder_input = torch.randint(0, self.config.vocab_size, (batch_size, decoder_seq_len)).to(
            self.device
        )
        decoder_target = torch.randint(0, self.config.vocab_size, (batch_size, decoder_seq_len)).to(
            self.device
        )

        # Forward and backward
        outputs = self.model(encoder_input, decoder_target, decoder_input_tokens=decoder_input)

        # Compute simple loss
        loss = outputs.mean()
        loss.backward()

        logger.info(f"  Loss value: {loss.item():.6f}")
        logger.info(f"  Input gradient exists: {encoder_input.grad is not None}")

        # Check that gradients exist for model parameters
        has_gradients = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                has_grad = param.grad is not None
                has_gradients.append(has_grad)
                if not has_grad:
                    logger.warning(f"    No gradient for: {name}")

        grad_ratio = sum(has_gradients) / len(has_gradients)
        logger.info(
            f"  Parameters with gradients: {sum(has_gradients)}/{len(has_gradients)} ({grad_ratio:.1%})"
        )

        self.assertGreater(sum(has_gradients), 0, "Some parameters should have gradients")

        # Put model back in eval mode
        self.model.eval()

        logger.info("  ✓ Gradient flow successful")

    def test_device_compatibility(self):
        """Test model works on available device."""
        logger.info("\n--- Test 1.3.8: Device Compatibility ---")

        logger.info(f"  Current device: {self.device}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        logger.info(f"  MPS available: {torch.backends.mps.is_available()}")

        # Check model is on correct device
        model_device = next(self.model.parameters()).device
        logger.info(f"  Model device: {model_device}")

        self.assertEqual(model_device.type, self.device.type, "Model should be on correct device")

        # Test forward pass on device
        batch_size = 1
        seq_len = 100
        decoder_seq_len = 50

        encoder_input = torch.randn(batch_size, seq_len, self.config.emb_dim).to(self.device)
        decoder_input = torch.randint(0, self.config.vocab_size, (batch_size, decoder_seq_len)).to(
            self.device
        )
        decoder_target = torch.randint(0, self.config.vocab_size, (batch_size, decoder_seq_len)).to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(encoder_input, decoder_target, decoder_input_tokens=decoder_input)

        # Check output is on same device
        self.assertEqual(outputs.device.type, self.device.type, "Output should be on same device")

        logger.info(f"  ✓ Device compatibility confirmed for {self.device}")


if __name__ == "__main__":
    logger.info("Starting Model Architecture Tests (Priority 1.3)")
    logger.info(f"Test output directory: {TEST_CONFIG['output_dir']}")
    logger.info(f"Log file: {TEST_CONFIG['output_dir']}/test_model_architecture.log")

    unittest.main(verbosity=2)
