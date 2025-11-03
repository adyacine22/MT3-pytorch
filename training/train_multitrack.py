"""
Multitrack Training Script for MT3-PyTorch.
Combines MAESTRO (piano) and SLAKH (multi-instrument) datasets.
"""

import sys
from pathlib import Path
import logging
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import random
import numpy as np
import threading
import time
from itertools import islice

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
warnings.filterwarnings("ignore", category=UserWarning, module="pydub")
warnings.filterwarnings("ignore", message="Empty filters detected in mel frequency basis")
warnings.filterwarnings("ignore", message=".*Xing stream.*")  # Catch all Xing-related warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.T5config import Magenta_T5Config
from config.training_config import get_config
from model.T5 import Transformer
from data.maestro_loader import MIDIDataset
from data.slakh_loader import SLAKHStemDataset, SLAKHMixDataset, SLAKHMixedDataset
from data.cached_maestro_loader import CachedMaestroDataset
from data.cached_slakh_loader import CachedSLAKHStemDataset, CachedSLAKHMixDataset, CachedSLAKHMixedDataset
from data.multitask_dataset import MultiTaskDataset
from data.audio_mixing import AudioMixingDataset
from data.training_utils import MaestroListDataset, ValListDataset, collate_batch
from data.constants import TOKEN_PAD, codec
from data.gpu_spectrogram import GPUSpectrogramComputer

# Setup basic logging (file handler will be added in main())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ExponentialMovingAverage:
    """
    Maintains exponential moving average of model parameters.
    
    EMA improves model stability and often leads to better performance.
    Formula: ema_param = decay * ema_param + (1 - decay) * model_param
    """
    
    def __init__(self, model, decay=0.9999):
        """
        Args:
            model: The model to track
            decay: Decay rate for EMA (typical values: 0.999, 0.9999)
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model):
        """Apply EMA parameters to model (for validation/inference)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        """Restore original model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Get EMA state for checkpointing."""
        return {'decay': self.decay, 'shadow': self.shadow}
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


def async_save_checkpoint(state_dict, path):
    """Save checkpoint asynchronously to avoid blocking training."""
    def _save():
        torch.save(state_dict, path)
    threading.Thread(target=_save, daemon=True).start()


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_fn(batch):
    """Collate function for batching."""
    return collate_batch(batch)


def setup_dataloaders(config):
    """Setup training and validation dataloaders.

    Args:
        config: Dictionary with dataset configuration

    Returns:
        tuple: (train_loader, val_loader)
    """
    logger.info("\nSetting up datasets...")

    datasets = {}
    val_datasets = {}
    
    use_cached = config.get("use_cached_data", False)
    
    # MAESTRO dataset (piano)
    if config.get("use_maestro", True):
        logger.info("  Loading MAESTRO dataset...")
        
        if use_cached:
            # Use cached dataset
            maestro_cache_dir = config.get("maestro_cache_dir")
            maestro_val_cache_dir = config.get("maestro_val_cache_dir")
            maestro_max_samples = config.get("maestro_max_samples")
            cache_size = config.get("cache_size")
            # Use None to allow dataset auto-detection; config can set True/False explicitly
            preload_all = config.get("preload_all", None)
            num_workers = config.get("num_workers", 0)  # Get num_workers for cache sizing
            
            maestro_dataset = CachedMaestroDataset(
                cache_dir=maestro_cache_dir,
                max_tracks=maestro_max_samples,
                cache_size=cache_size,
                preload_all=preload_all,
                num_workers=max(1, num_workers)  # Pass to dataset for RAM allocation
            )
            datasets["maestro"] = maestro_dataset
            logger.info(f"  ✓ MAESTRO train: {len(datasets['maestro'])} samples (cached)")
            
            # Validation
            if maestro_val_cache_dir:
                val_max = config.get("val_max_samples", 100)
                # Validation datasets are small; avoid auto-preloading by default.
                # Allow explicit override via config key `preload_all_val` (default False).
                maestro_val_dataset = CachedMaestroDataset(
                    cache_dir=maestro_val_cache_dir,
                    max_tracks=val_max,
                    cache_size=cache_size,
                    preload_all=config.get("preload_all_val", False),
                    num_workers=1  # Validation uses fewer workers
                )
                val_datasets["maestro"] = maestro_val_dataset
                logger.info(f"  ✓ MAESTRO val: {len(val_datasets['maestro'])} samples (cached)")
        else:
            # Use raw dataset
            try:
                maestro_train = MIDIDataset(split="train")
                maestro_samples = []
                maestro_max_samples = config.get("maestro_max_samples", 100)
                logger.info("    Loading MAESTRO samples...")
                for i, sample in enumerate(maestro_train):
                    maestro_samples.append(sample)
                    if i >= maestro_max_samples - 1:
                        break
                    if (i + 1) % 20 == 0:
                        logger.info(f"Loaded {i + 1} MAESTRO samples")
                datasets["maestro"] = MaestroListDataset(maestro_samples)
                logger.info(f"  ✓ MAESTRO: {len(datasets['maestro'])} samples")
            except Exception as e:
                logger.warning(f"Could not load MAESTRO: {e}")

    # SLAKH Stem Dataset (individual instruments)
    if config.get("use_slakh_stems", True):
        logger.info("  Loading SLAKH stems dataset...")
        try:
            if use_cached:
                slakh_cache_dir = config.get("slakh_cache_dir")
                slakh_val_cache_dir = config.get("slakh_val_cache_dir")
                slakh_max_tracks = config.get("slakh_max_tracks")
                cache_size = config.get("cache_size")
                preload_all = config.get("preload_all", None)
                num_workers = config.get("num_workers", 0)
                
                slakh_stems = CachedSLAKHStemDataset(
                    cache_dir=slakh_cache_dir,
                    max_tracks=slakh_max_tracks,
                    cache_size=cache_size,
                    preload_all=preload_all,
                    num_workers=max(1, num_workers)
                )
                datasets["slakh_stems"] = slakh_stems
                logger.info(f"  ✓ SLAKH stems train: {len(datasets['slakh_stems'])} samples (cached)")
                
                # Validation
                if slakh_val_cache_dir:
                    val_max = config.get("val_max_samples", 100)
                    slakh_val_stems = CachedSLAKHStemDataset(
                        cache_dir=slakh_val_cache_dir,
                        max_tracks=val_max,
                        cache_size=cache_size,
                        preload_all=config.get("preload_all_val", False),
                        num_workers=1
                    )
                    val_datasets["slakh_stems"] = slakh_val_stems
                    logger.info(f"  ✓ SLAKH stems val: {len(val_datasets['slakh_stems'])} samples (cached)")
            else:
                slakh_stems = SLAKHStemDataset(
                    split="train", max_tracks=config.get("slakh_max_tracks", 10)
                )
                datasets["slakh_stems"] = slakh_stems
                logger.info(f"  ✓ SLAKH stems: {len(datasets['slakh_stems'])} samples")
        except Exception as e:
            logger.warning(f"Could not load SLAKH stems: {e}")

    # SLAKH Mix Dataset (full tracks with all instruments)
    if config.get("use_slakh_mix", False):
        logger.info("  Loading SLAKH mix dataset...")
        try:
            if use_cached:
                slakh_cache_dir = config.get("slakh_cache_dir")
                slakh_val_cache_dir = config.get("slakh_val_cache_dir")
                slakh_max_tracks = config.get("slakh_max_tracks")
                cache_size = config.get("cache_size")
                preload_all = config.get("preload_all", None)
                num_workers = config.get("num_workers", 0)
                
                slakh_mix = CachedSLAKHMixDataset(
                    cache_dir=slakh_cache_dir,
                    max_tracks=slakh_max_tracks,
                    cache_size=cache_size,
                    preload_all=preload_all,
                    num_workers=max(1, num_workers)
                )
                datasets["slakh_mix"] = slakh_mix
                logger.info(f"  ✓ SLAKH mix train: {len(datasets['slakh_mix'])} samples (cached)")
                # Validation
                if slakh_val_cache_dir:
                    val_max = config.get("val_max_samples", 100)
                    slakh_val_mix = CachedSLAKHMixDataset(
                        cache_dir=slakh_val_cache_dir,
                        max_tracks=val_max,
                        cache_size=cache_size,
                        preload_all=config.get("preload_all_val", False),
                        num_workers=1
                    )
                    val_datasets["slakh_mix"] = slakh_val_mix
                    logger.info(f"  ✓ SLAKH mix val: {len(val_datasets['slakh_mix'])} samples (cached)")
            else:
                slakh_mix = SLAKHMixDataset(
                    split="train", max_tracks=config.get("slakh_max_tracks", 10)
                )
                datasets["slakh_mix"] = slakh_mix
                logger.info(f"  ✓ SLAKH mix: {len(datasets['slakh_mix'])} samples")
        except Exception as e:
            logger.warning(f"Could not load SLAKH mix: {e}")

    # SLAKH Mixed Dataset (random stem combinations)
    if config.get("use_slakh_mixed", False):
        logger.info("  Loading SLAKH mixed dataset...")
        try:
            if use_cached:
                slakh_cache_dir = config.get("slakh_cache_dir")
                slakh_val_cache_dir = config.get("slakh_val_cache_dir")
                slakh_max_tracks = config.get("slakh_max_tracks")
                cache_size = config.get("cache_size")
                preload_all = config.get("preload_all", None)
                
                slakh_mixed = CachedSLAKHMixedDataset(
                    cache_dir=slakh_cache_dir,
                    max_tracks=slakh_max_tracks,
                    cache_size=cache_size,
                    preload_all=preload_all
                )
                datasets["slakh_mixed"] = slakh_mixed
                logger.info(f"  ✓ SLAKH mixed train: {len(datasets['slakh_mixed'])} samples (cached)")
                
                # Validation
                if slakh_val_cache_dir:
                    val_max = config.get("val_max_samples", 100)
                    slakh_val_mixed = CachedSLAKHMixedDataset(
                        cache_dir=slakh_val_cache_dir,
                        max_tracks=val_max,
                        cache_size=cache_size,
                        preload_all=config.get("preload_all_val", False)
                    )
                    val_datasets["slakh_mixed"] = slakh_val_mixed
                    logger.info(f"  ✓ SLAKH mixed val: {len(val_datasets['slakh_mixed'])} samples (cached)")
            else:
                slakh_mixed = SLAKHMixedDataset(
                    split="train",
                    max_tracks=config.get("slakh_max_tracks", 10),
                    min_stems=config.get("min_stems", 1),
                    max_stems=config.get("max_stems", 4),
                )
                datasets["slakh_mixed"] = slakh_mixed
                logger.info(f"  ✓ SLAKH mixed: {len(datasets['slakh_mixed'])} samples")
        except Exception as e:
            logger.warning(f"Could not load SLAKH mixed: {e}")

    if not datasets:
        raise ValueError("No datasets loaded successfully!")

    # Create multi-task dataset with temperature sampling
    logger.info("\n  Creating multi-task dataset...")
    temperatures = config.get("temperatures", {})
    train_dataset = MultiTaskDataset(datasets=datasets, temperatures=temperatures)
    logger.info(f"  ✓ Multi-task dataset: {len(train_dataset)} total samples")
    logger.info(f"  Temperature settings: {temperatures}")
    
    # Apply audio mixing if enabled (for pretraining)
    if config.get("use_audio_mixing", False):
        max_mix = config.get("max_examples_per_mix", 8)
        mix_prob = config.get("mix_probability", 1.0)
        logger.info(f"\n  Applying audio mixing wrapper...")
        logger.info(f"    Max examples per mix: {max_mix}")
        logger.info(f"    Mix probability: {mix_prob}")
        train_dataset = AudioMixingDataset(
            base_dataset=train_dataset,
            max_examples_per_mix=max_mix,
            mix_probability=mix_prob
        )
        logger.info(f"  ✓ Audio mixing enabled: up to {max_mix} examples per mix")

    # Create validation dataset from val_datasets
    logger.info("\n  Setting up validation data...")
    val_dataset = None
    
    if val_datasets:
        # Create multi-task validation dataset
        val_dataset = MultiTaskDataset(datasets=val_datasets, temperatures=temperatures)
        logger.info(f"  ✓ Validation: {len(val_dataset)} samples")
    else:
        logger.warning("  ! No validation datasets loaded")

    # Create dataloaders
    num_workers = config.get("num_workers", 0)
    pin_memory = config.get("pin_memory", True)
    prefetch_factor = config.get("prefetch_factor", 2) if num_workers > 0 else None
    persistent_workers = config.get("persistent_workers", False) and num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get("batch_size", 4),
            shuffle=False,
            num_workers=min(num_workers, 2),  # Fewer workers for validation
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )

    return train_loader, val_loader


def train_steps(model, train_loader, optimizer, criterion, device, config, scaler=None, 
                scheduler=None, start_step=0, val_loader=None, checkpoint_dir=None, config_name=None, ema=None):
    """Train for a specified number of steps with optional mixed precision and gradient accumulation.
    
    HYBRID APPROACH: Computes mel-spectrograms on GPU from frames loaded by DataLoader.
    This matches legacy MT3 pipeline and eliminates CPU bottleneck.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        config: Training configuration
        scaler: GradScaler for mixed precision
        scheduler: Learning rate scheduler
        start_step: Starting step number
        val_loader: Validation DataLoader
        checkpoint_dir: Directory to save checkpoints
        config_name: Name of configuration (for checkpoint naming)
        
    Returns:
        final_step: The final step number reached
    """
    model.train()
    total_loss = 0
    num_steps = 0
    max_steps = config.get("max_steps", 1000)
    log_interval = config.get("log_interval", 10)
    val_interval = config.get("val_interval", 100)
    save_interval = config.get("save_interval", 500)
    use_mixed_precision = config.get("use_mixed_precision", False) and scaler is not None
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    
    # Create GPU spectrogram computer (hybrid approach)
    logger.info(f"  Initializing GPU spectrogram computer...")
    spec_computer = GPUSpectrogramComputer(device=device)
    logger.info(f"  ✓ GPU spectrogram computer ready")
    
    best_val_loss = float("inf")
    current_step = start_step

    logger.info(f"\n{'='*80}")
    logger.info(f"STEP-BASED TRAINING")
    logger.info(f"  Starting from step: {start_step}")
    logger.info(f"  Target steps: {max_steps}")
    logger.info(f"  Batch size: {config.get('batch_size', 4)}")
    logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {config.get('batch_size', 4) * gradient_accumulation_steps}")
    if use_mixed_precision:
        logger.info(f"  Mixed Precision: ENABLED (FP16)")
    logger.info(f"{'='*80}")

    # Create infinite iterator from train_loader for step-based training
    # Proper infinite loader that reshuffles each epoch (not itertools.cycle!)
    def infinite_dataloader(loader):
        """Infinite data loader that properly reshuffles each epoch."""
        while True:
            for batch in loader:
                yield batch
    
    train_iter = infinite_dataloader(train_loader)
    logger.info("\n  ✓ Infinite data iterator created (reshuffles each epoch)")
    
    import time
    accumulated_loss = 0.0
    
    logger.info("\n🚀 Starting training loop...")
    for step in range(start_step, max_steps):
        batch_start = time.time()
        
        # Log first batch loading
        if step == 0:
            logger.info(f"  ⏳ Loading first batch (this may take 10-60s due to DataLoader worker initialization + torch.compile)...")
        
        # Track timing for each phase
        time_fetch = 0
        time_to_device = 0
        time_spec_compute = 0  # NEW: Time for GPU spectrogram computation
        time_forward = 0
        time_backward = 0
        
        # Accumulate gradients over multiple micro-batches
        for accum_step in range(gradient_accumulation_steps):
            # Fetch batch - log every accumulation step
            fetch_start = time.time()
            logger.info(f"     [{step}] Fetching batch from DataLoader (accum_step {accum_step+1}/{gradient_accumulation_steps})...")
            
            batch = next(train_iter)
            fetch_time = time.time() - fetch_start
            time_fetch += fetch_time
            
            logger.info(f"     [{step}] ✓ Batch fetched in {fetch_time:.3f}s")
            
            if step < 3 or (step < 100 and step % 10 == 0):
                logger.info(f"     [{step}] Moving data to device {device}...")
            
            # Move FRAMES to device (not mel-spectrograms yet)
            device_start = time.time()
            frames = batch["inputs"].to(device)  # Shape: [B, 256, 128]
            targets = batch["targets"].to(device)
            device_time = time.time() - device_start
            time_to_device += device_time
            
            if step < 3 or (step < 100 and step % 10 == 0):
                logger.info(f"     [{step}] ✓ Data moved to device in {device_time:.3f}s (frames: {frames.shape}, targets: {targets.shape})")
            
            # Compute mel-spectrograms on GPU (HYBRID APPROACH)
            # This matches legacy MT3: frames → flatten → mel-spectrogram
            spec_start = time.time()
            inputs = spec_computer(frames)  # [B, 256, 128] -> [B, 512, 256]
            spec_time = time.time() - spec_start
            time_spec_compute += spec_time
            
            if step < 3 or (step < 100 and step % 10 == 0):
                logger.info(f"     [{step}] ✓ Mel-spectrograms computed on GPU in {spec_time:.3f}s (shape: {inputs.shape})")

            # Ensure targets are long type for embedding
            if targets.dtype != torch.long:
                targets = targets.long()

            if step < 3 or (step < 100 and step % 10 == 0):
                logger.info(f"     [{step}] Starting forward pass...")
            
            # Forward pass with optional mixed precision
            forward_start = time.time()
            if use_mixed_precision:
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs, targets[:, :-1])
                    loss = criterion(outputs.permute(0, 2, 1), targets[:, 1:])
                    # Scale loss by accumulation steps
                    loss = loss / gradient_accumulation_steps
            else:
                # Standard FP32 training
                outputs = model(inputs, targets[:, :-1])
                loss = criterion(outputs.permute(0, 2, 1), targets[:, 1:])
                # Scale loss by accumulation steps
                loss = loss / gradient_accumulation_steps
            
            forward_time = time.time() - forward_start
            time_forward += forward_time
            
            # Log forward pass completion with loss for every accumulation step
            logger.info(f"     [{step}] ✓ Forward pass completed in {forward_time:.3f}s, loss = {loss.item() * gradient_accumulation_steps:.4f} (accum_step {accum_step+1}/{gradient_accumulation_steps})")
            
            if step < 3 or (step < 100 and step % 10 == 0):
                logger.info(f"     [{step}] Starting backward pass...")

            # Backward pass
            backward_start = time.time()
            if use_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            backward_time = time.time() - backward_start
            time_backward += backward_time
            
            if step < 3 or (step < 100 and step % 10 == 0):
                logger.info(f"     [{step}] ✓ Backward pass completed in {backward_time:.3f}s")
            
            accumulated_loss += loss.item()
        
        # Update weights after accumulating gradients
        if step < 3 or (step < 100 and step % 10 == 0):
            logger.info(f"     [{step}] Gradient clipping and optimizer step...")
        
        optim_start = time.time()
        grad_norm = None
        if use_mixed_precision:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get("gradient_clip_norm", 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get("gradient_clip_norm", 1.0))
            optimizer.step()
        
        # Update EMA after optimizer step
        if ema is not None:
            ema.update(model)
        
        optimizer.zero_grad(set_to_none=True)  # Use set_to_none=True to free memory
        
        # Clear CUDA cache periodically to prevent memory fragmentation
        if current_step % 50 == 0:
            torch.cuda.empty_cache()
        
        optim_time = time.time() - optim_start
        
        if step < 3 or (step < 100 and step % 10 == 0):
            logger.info(f"     [{step}] ✓ Optimizer step completed in {optim_time:.3f}s, grad_norm = {grad_norm:.4f}")
        
        # Update counters
        current_step = step + 1
        num_steps += 1
        total_loss += accumulated_loss
        
        # Update learning rate scheduler (step-based)
        if scheduler is not None:
            scheduler.step()

        # Calculate total step time
        step_time = time.time() - batch_start
        
        # Log progress with timing breakdown
        if current_step % log_interval == 0:
            avg_loss = total_loss / num_steps
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            
            logger.info(
                f"  Step {current_step}/{max_steps}: loss = {accumulated_loss:.4f}, avg_loss = {avg_loss:.4f}, "
                f"lr = {lr:.6f}, grad_norm = {grad_norm:.4f}"
            )
            logger.info(
                f"    ⏱️  Timing: total={step_time:.3f}s (fetch={time_fetch:.3f}s, "
                f"to_device={time_to_device:.3f}s, spec_compute={time_spec_compute:.3f}s, "
                f"forward={time_forward:.3f}s, backward={time_backward:.3f}s, optim={optim_time:.3f}s)"
            )
        
        # Reset accumulated loss for next step
        accumulated_loss = 0.0
        
        # Validate periodically
        if val_loader is not None and current_step % val_interval == 0:
            logger.info(f"\n  🔍 Starting validation at step {current_step}...")
            val_start = time.time()
            
            # Use EMA weights for validation if available
            if ema is not None:
                logger.info(f"     Applying EMA weights for validation...")
                ema.apply_shadow(model)
            
            val_loss = validate(model, val_loader, criterion, device, config)
            val_time = time.time() - val_start
            
            # Restore training weights
            if ema is not None:
                logger.info(f"     Restoring training weights...")
                ema.restore(model)
            
            model.train()  # Back to training mode
            logger.info(f"  ✓ Validation completed in {val_time:.2f}s")
            
            # Save best model
            if checkpoint_dir is not None and val_loss < best_val_loss:
                logger.info(f"     🎯 New best validation loss: {val_loss:.4f} (previous: {best_val_loss:.4f})")
                best_val_loss = val_loss
                checkpoint_path = checkpoint_dir / f"best_{config_name}_model.pt"
                logger.info(f"     Saving best model checkpoint...")
                async_save_checkpoint(
                    {
                        "step": current_step,
                        "model_state_dict": model.state_dict(),
                        "ema_state_dict": ema.state_dict() if ema else None,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                        "scaler_state_dict": scaler.state_dict() if scaler else None,
                        "train_loss": total_loss / num_steps,
                        "val_loss": val_loss,
                        "config": config,
                        "config_name": config_name,
                    },
                    checkpoint_path,
                )
                logger.info(f"  ✓ Saved best model to {checkpoint_path} (async)")
            else:
                logger.info(f"     Validation loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
        
        # Save periodic checkpoints
        if checkpoint_dir is not None and current_step % save_interval == 0:
            logger.info(f"\n  💾 Saving periodic checkpoint at step {current_step}...")
            checkpoint_path = checkpoint_dir / f"{config_name}_step_{current_step}.pt"
            checkpoint_data = {
                "step": current_step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict() if ema else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "train_loss": total_loss / num_steps,
                "config": config,
                "config_name": config_name,
            }
            logger.info(f"     Checkpoint size: ~{sum(p.numel() for p in model.parameters()) * 4 / 1024**2:.1f} MB")
            async_save_checkpoint(checkpoint_data, checkpoint_path)
            logger.info(f"  ✓ Checkpoint saved to {checkpoint_path} (async)")

    avg_loss = total_loss / num_steps if num_steps > 0 else 0
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING SUMMARY:")
    logger.info(f"  Steps Completed: {num_steps}")
    logger.info(f"  Average Loss: {avg_loss:.4f}")
    logger.info(f"  Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"{'='*80}")

    return current_step


def validate(model, val_loader, criterion, device, config):
    """Validate the model with GPU spectrogram computation (hybrid approach)."""
    if val_loader is None:
        return float("inf")

    logger.info(f"     Running validation...")
    model.eval()
    total_loss = 0
    num_batches = 0
    max_batches = config.get("max_val_batches", config.get("val_max_samples", 20))
    
    # Create GPU spectrogram computer for validation
    spec_computer = GPUSpectrogramComputer(device=device)

    with torch.no_grad():
        # Use islice to limit batches without fetching unused batches
        for batch_idx, batch in enumerate(islice(val_loader, max_batches)):
            # Move frames to device and compute mel-spectrograms
            frames = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            
            # Compute mel-spectrograms on GPU
            inputs = spec_computer(frames)

            # Ensure targets are long type for embedding
            if targets.dtype != torch.long:
                targets = targets.long()

            outputs = model(inputs, targets[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), targets[:, 1:])

            total_loss += loss.item()
            num_batches += 1
            
            # Log progress for first few and every 5th batch
            if batch_idx < 3 or (batch_idx + 1) % 5 == 0:
                logger.info(f"       Validation batch {batch_idx + 1}/{max_batches}: loss = {loss.item():.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    logger.info(f"     ✓ Validation complete: avg_loss = {avg_loss:.4f} ({num_batches} batches)")

    return avg_loss


def main(config_name="A100_pretraining_test"):
    """Main training function (step-based training for data mixing).

    Args:
        config_name: Name of configuration to use. Options:
            - 'A100_pretraining_full': Full pretraining (165k steps, batch=1024, mixing ON)
            - 'A100_pretraining_test': Test pretraining (500 steps, batch=1024, mixing ON)
            - 'A100_finetuning_full': Full finetuning (50k steps, batch=256, mixing OFF)
            - 'A100_finetuning_test': Test finetuning (500 steps, batch=256, mixing OFF)
    """
    # Load configuration first
    try:
        config = get_config(config_name)
    except ValueError as e:
        print(f"Error loading config: {e}")
        print("Available configs: A100_pretraining_full, A100_pretraining_test, A100_finetuning_full, A100_finetuning_test")
        return
    
    # Setup file logging now that we have config
    log_dir = Path(config.get("log_dir", Path(__file__).parent.parent / "test_outputs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_multitrack.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("MULTITRACK TRAINING - MAESTRO + SLAKH")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config_name}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Get device
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Enable Flash Attention and memory-efficient attention backends (PyTorch 2.0+)
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("  ✓ Flash Attention backends enabled")
    
    # Enable TensorFloat32 for A100 GPU (20% speedup for matrix ops)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # Enables TF32 on A100
        logger.info("  ✓ TensorFloat32 enabled for faster matmul")

    # Log configuration details
    logger.info(f"\nLoaded config: {config_name}")
    logger.info(
        f"  Datasets: MAESTRO={config.get('use_maestro', False)}, "
        f"SLAKH_stems={config.get('use_slakh_stems', False)}, "
        f"SLAKH_mix={config.get('use_slakh_mix', False)}, "
        f"SLAKH_mixed={config.get('use_slakh_mixed', False)}"
    )
    logger.info(f"  Batch size: {config['batch_size']}, Steps: {config.get('max_steps', 1000)}")
    logger.info(f"  Validation interval: {config.get('val_interval', 100)} steps")
    logger.info(f"  Log interval: {config.get('log_interval', 10)} steps")

    # Initialize model
    logger.info("\nInitializing model...")
    model_config = Magenta_T5Config()
    use_flash_attention = config.get("use_flash_attention", True)
    model = Transformer(config=model_config, use_flash_attention=use_flash_attention)
    model = model.to(device)
    
    if use_flash_attention:
        logger.info("  ✓ Flash Attention enabled in model")

    total_params = count_parameters(model)
    logger.info(f"  Total parameters: {total_params:,}")
    
    # Initialize EMA (Exponential Moving Average) for more stable inference
    use_ema = config.get("use_ema", True)
    ema_decay = config.get("ema_decay", 0.9999)
    if use_ema:
        ema = ExponentialMovingAverage(model, decay=ema_decay)
        logger.info(f"  ✓ EMA initialized (decay={ema_decay})")
    else:
        ema = None
        logger.info("  EMA disabled")

    # Apply torch.compile() for faster execution (PyTorch 2.0+)
    use_torch_compile = config.get("use_torch_compile", False)
    if use_torch_compile and hasattr(torch, 'compile'):
        logger.info("\n⚡ Compiling model with torch.compile()...")
        compile_mode = config.get("compile_mode", "default")
        try:
            model = torch.compile(model, mode=compile_mode)
            logger.info(f"  ✓ Model compiled (mode={compile_mode})")
            logger.info("  Note: First batch will be slower due to compilation")
        except Exception as e:
            logger.warning(f"  Warning: torch.compile() failed: {e}")
            logger.info("  Continuing without compilation")
    elif use_torch_compile:
        logger.warning("  torch.compile() requested but not available (requires PyTorch 2.0+)")

    # Setup datasets and loaders
    train_loader, val_loader = setup_dataloaders(config)

    # Setup optimizer and loss (YourMT3+ configuration)
    use_fused = config.get("use_fused_optimizer", True) and torch.cuda.is_available()
    optimizer_betas = config.get("optimizer_betas", (0.9, 0.98))  # YourMT3+ uses (0.9, 0.98)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        betas=optimizer_betas,
        weight_decay=config.get("weight_decay", 0.1),  # YourMT3+ uses 0.1
        fused=use_fused,  # Fused AdamW for 10-15% speedup on CUDA
    )
    logger.info(f"  Optimizer: AdamW (betas={optimizer_betas}, weight_decay={config.get('weight_decay', 0.1)})")
    if use_fused:
        logger.info("  Using Fused AdamW optimizer (faster on CUDA)")
    
    # Loss function with label smoothing (LEGACY: 0.1)
    label_smoothing = config.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, label_smoothing=label_smoothing)
    if label_smoothing > 0:
        logger.info(f"  Label smoothing: {label_smoothing}")

    # Initialize mixed precision scaler if enabled
    scaler = None
    use_mixed_precision = config.get("use_mixed_precision", False)
    if use_mixed_precision:
        if device.type == "cuda":
            scaler = GradScaler(device="cuda")
            logger.info("\n  Mixed Precision: ENABLED (FP16 with GradScaler)")
        else:
            logger.warning(
                f"\n  Mixed Precision requested but device is '{device.type}' (not CUDA). "
                "Falling back to FP32 training."
            )
            use_mixed_precision = False

    # Learning rate scheduler with warmup (step-based)
    warmup_steps = config.get("warmup_steps", 0)
    max_steps = config.get("max_steps", 1000)
    scheduler_type = config.get("scheduler_type", "constant")  # LEGACY: "constant"
    
    if scheduler_type == "constant":
        # Constant LR (LEGACY MT3 uses this!)
        if warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=warmup_steps
            )
            constant_scheduler = ConstantLR(optimizer, factor=1.0)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, constant_scheduler],
                milestones=[warmup_steps]
            )
            logger.info(f"\n  Learning Rate Scheduler: Warmup ({warmup_steps} steps) + Constant")
        else:
            scheduler = None
            logger.info(f"\n  Learning Rate: Constant (no scheduler)")
    elif scheduler_type == "inverse_sqrt":
        # InverseSqrt scheduler (alternative)
        from training.schedulers import InverseSqrtWithWarmup
        scheduler = InverseSqrtWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            base_lr=config.get("learning_rate", 1e-3),
            min_lr=1e-6
        )
        logger.info(f"\n  Learning Rate Scheduler: InverseSqrt with Warmup ({warmup_steps} steps)")
    else:
        # Cosine scheduler with warmup
        from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
        
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.01,
                total_iters=warmup_steps
            )
            main_scheduler = CosineAnnealingLR(
                optimizer, T_max=(max_steps - warmup_steps), eta_min=1e-5
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
            logger.info(f"\n  Learning Rate Scheduler: Warmup ({warmup_steps} steps) + CosineAnnealing")
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)
            logger.info(f"\n  Learning Rate Scheduler: CosineAnnealing ({max_steps} steps)")
            logger.info(f"\n  Learning Rate Scheduler: CosineAnnealing ({max_steps} steps)")

    logger.info("\nStarting training...")
    logger.info(f"  Total steps: {max_steps}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Learning rate: {config.get('learning_rate', 1e-4)}")
    logger.info(f"  Weight decay: {config.get('weight_decay', 0.01)}")
    logger.info(f"  Log interval: {config.get('log_interval', 10)} steps")
    logger.info(f"  Validation interval: {config.get('val_interval', 100)} steps")
    logger.info(f"  Save interval: {config.get('save_interval', 500)} steps")

    checkpoint_dir = Path(config.get("checkpoint_dir", Path(__file__).parent.parent / "checkpoint"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Step-based training loop
    final_step = train_steps(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        scaler=scaler,
        scheduler=scheduler,
        start_step=0,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        config_name=config_name,
        ema=ema
    )

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Configuration: {config_name}")
    logger.info(f"Final step: {final_step}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    log_file = Path(config.get("log_dir", Path(__file__).parent.parent / "test_outputs")) / "train_multitrack.log"
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MT3 multitrack model")
    parser.add_argument(
        "--config",
        type=str,
        default="A100_pretraining_test",
        choices=[
            "A100_pretraining_full",
            "A100_pretraining_test",
            "A100_finetuning_full",
            "A100_finetuning_test",
        ],
        help="Training configuration to use (default: A100_pretraining_test)",
    )

    args = parser.parse_args()
    main(args.config)
