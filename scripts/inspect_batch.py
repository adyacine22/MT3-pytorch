"""
Inspect a single batch from the dataloader for a given configuration.
This helps verify the output of the dataloaders and check data types, shapes, and ranges.

Usage:
    python scripts/inspect_batch.py --config A100_pretraining_test
"""

import sys
from pathlib import Path
import torch
import argparse
import logging

# Suppress verbose logging from other modules
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("inspect_batch")
logger.setLevel(logging.INFO)

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from config.training_config import get_config
from training.train_multitrack import setup_dataloaders

def inspect_batch(config_name="A100_pretraining_test"):
    """
    Sets up dataloader for a given config, fetches one batch, and prints its structure.
    """
    logger.info(f"--- Inspecting batch for config: {config_name} ---")

    # 1. Get configuration
    try:
        config = get_config(config_name)
    except ValueError as e:
        logger.error(f"Error loading config: {e}")
        return
        
    # Use a small number of workers for inspection to avoid CUDA init issues in main process
    # before we are ready.
    config['num_workers'] = 2
    config['persistent_workers'] = False
    # Use a smaller batch size for easier inspection
    config['batch_size'] = 4

    # 2. Setup dataloaders
    try:
        # Redirect logging to hide verbose setup details unless there's an error
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        
        train_loader, val_loader = setup_dataloaders(config)
        
        logging.getLogger().setLevel(original_level)
        logger.info("--- DataLoader setup complete ---")

    except Exception as e:
        logging.getLogger().setLevel(original_level)
        logger.error(f"Error setting up dataloaders: {e}", exc_info=True)
        return

    if train_loader is None:
        logger.error("Train loader is None, cannot inspect batch.")
        return

    # 3. Fetch one batch
    try:
        logger.info("--- Fetching one batch from train_loader ---")
        batch = next(iter(train_loader))
        logger.info("--- Batch fetched successfully ---")
    except Exception as e:
        logger.error(f"Error fetching batch: {e}", exc_info=True)
        return

    # 4. Print detailed information
    logger.info("\n--- Batch Content Inspection ---")
    if isinstance(batch, dict):
        for key, value in batch.items():
            logger.info(f"\nKey: '{key}'")
            if isinstance(value, torch.Tensor):
                logger.info(f"  Type: torch.Tensor")
                logger.info(f"  Shape: {value.shape}")
                logger.info(f"  Dtype: {value.dtype}")
                logger.info(f"  Device: {value.device}")
                try:
                    logger.info(f"  Min value: {value.min()}")
                    logger.info(f"  Max value: {value.max()}")
                    logger.info(f"  Mean value: {value.float().mean()}")
                except RuntimeError as e:
                    logger.warning(f"  Could not compute stats (likely empty tensor): {e}")
                logger.info(f"  Requires grad: {value.requires_grad}")
            else:
                logger.info(f"  Type: {type(value)}")
                if isinstance(value, (list, tuple)):
                    logger.info(f"  Length: {len(value)}")
                    if len(value) > 0:
                        logger.info(f"  First element type: {type(value[0])}")
    else:
        logger.info(f"Batch type: {type(batch)}")
    
    logger.info("\n--- Inspection complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a single batch from the dataloader.")
    parser.add_argument(
        "--config",
        type=str,
        default="A100_pretraining_test",
        help="Training configuration to use.",
    )
    args = parser.parse_args()
    inspect_batch(args.config)