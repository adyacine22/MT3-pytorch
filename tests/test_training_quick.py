#!/usr/bin/env python3
"""
Quick test to verify training pipeline is fully operational.
Tests:
1. Model initialization
2. Data loading with TIE + RLE
3. Forward pass
4. Backward pass
5. Optimizer step
"""

import sys
sys.path.insert(0, '/cluster/home/ahmedyz/yassine/MT3-pytorch')

import torch
import torch.nn as nn
from pathlib import Path
from config.T5config import Magenta_T5Config
from model.T5 import Transformer
from data.cached_maestro_loader import CachedMaestroDataset
from torch.utils.data import DataLoader
from data.training_utils import collate_batch
from data.constants import TOKEN_PAD

def test_training_pipeline():
    """Test the complete training pipeline end-to-end."""
    
    print("="*70)
    print("TRAINING PIPELINE TEST")
    print("="*70)
    
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # 2. Model
    print("\n📦 Initializing model...")
    model_config = Magenta_T5Config()
    model = Transformer(config=model_config, use_flash_attention=False)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model initialized: {total_params:,} parameters")
    
    # 3. Dataset
    print("\n📁 Loading dataset...")
    cache_dir = "/cluster/home/ahmedyz/yassine/MT3-pytorch/cache/maestro/train"
    
    if not Path(cache_dir).exists():
        print(f"  ❌ Cache directory not found: {cache_dir}")
        print("  Run preprocessing first!")
        return False
    
    dataset = CachedMaestroDataset(cache_dir=cache_dir, max_tracks=5)
    print(f"  ✓ Dataset loaded: {len(dataset)} tracks")
    
    # 4. DataLoader
    print("\n🔄 Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch
    )
    print(f"  ✓ DataLoader created (batch_size=2)")
    
    # 5. Optimizer & Loss
    print("\n⚙️  Setting up optimizer and loss...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    print(f"  ✓ Optimizer: AdamW (lr=1e-4)")
    print(f"  ✓ Loss: CrossEntropyLoss (ignore_index={TOKEN_PAD})")
    
    # 6. Training loop (2 steps)
    print("\n🚀 Running training steps...")
    model.train()
    
    for step, batch in enumerate(dataloader):
        if step >= 2:  # Just 2 steps
            break
        
        print(f"\n  Step {step + 1}/2:")
        
        # Move to device
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        
        print(f"    Input shape: {inputs.shape}")
        print(f"    Target shape: {targets.shape}")
        
        # Ensure targets are long type
        if targets.dtype != torch.long:
            targets = targets.long()
        
        # Forward pass
        try:
            outputs = model(inputs, targets[:, :-1])
            print(f"    Output shape: {outputs.shape}")
            print(f"    ✓ Forward pass successful")
        except Exception as e:
            print(f"    ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Compute loss
        try:
            loss = criterion(outputs.permute(0, 2, 1), targets[:, 1:])
            print(f"    Loss: {loss.item():.4f}")
            print(f"    ✓ Loss computation successful")
        except Exception as e:
            print(f"    ❌ Loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Backward pass
        try:
            optimizer.zero_grad()
            loss.backward()
            print(f"    ✓ Backward pass successful")
        except Exception as e:
            print(f"    ❌ Backward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Optimizer step
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            print(f"    Gradient norm: {grad_norm:.4f}")
            print(f"    ✓ Optimizer step successful")
        except Exception as e:
            print(f"    ❌ Optimizer step failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*70}")
    print("✅ TRAINING PIPELINE IS FULLY OPERATIONAL!")
    print(f"{'='*70}")
    print("\nAll components working:")
    print("  ✓ Model initialization")
    print("  ✓ Data loading (with TIE + RLE)")
    print("  ✓ Forward pass")
    print("  ✓ Loss computation")
    print("  ✓ Backward pass")
    print("  ✓ Gradient clipping")
    print("  ✓ Optimizer step")
    print(f"\n{'='*70}")
    
    return True


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    success = test_training_pipeline()
    sys.exit(0 if success else 1)
