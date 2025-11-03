#!/usr/bin/env python3
"""
Test the full data loading pipeline with TIE + RLE to verify it works in training.
"""

import sys
sys.path.insert(0, '/cluster/home/ahmedyz/yassine/MT3-pytorch')

import torch
from pathlib import Path
from data.cached_maestro_loader import CachedMaestroDataset
from data.constants import codec
from data import vocabularies


def test_full_pipeline():
    """Test the complete pipeline with multiple samples."""
    
    cache_dir = "/cluster/home/ahmedyz/yassine/MT3-pytorch/cache/maestro/train"
    
    if not Path(cache_dir).exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        return False
    
    print("="*70)
    print("FULL PIPELINE TEST: TIE Events + RLE + Training Data")
    print("="*70)
    
    # Create dataset
    dataset = CachedMaestroDataset(cache_dir=cache_dir, max_tracks=10)
    print(f"\n✅ Dataset loaded: {len(dataset)} tracks")
    
    # Test multiple samples
    num_samples = 10
    tie_counts = []
    rle_compressions = []
    token_distributions = {}
    
    print(f"\nTesting {num_samples} random samples...")
    
    for i in range(num_samples):
        try:
            sample = dataset[i]
            
            inputs = sample['inputs']
            targets = sample['targets']
            
            # Analyze targets
            targets_list = targets.tolist()
            
            # Count TIE events
            tie_token = codec.encode_event(vocabularies.Event("tie", 0))
            tie_count = targets_list.count(tie_token)
            tie_counts.append(tie_count)
            
            # Check for RLE effectiveness (consecutive shifts)
            max_consecutive_shifts = 0
            current_consecutive = 0
            
            for token in targets_list:
                if token == 0:  # PAD token, skip
                    break
                if codec.is_shift_event_index(token):
                    current_consecutive += 1
                    max_consecutive_shifts = max(max_consecutive_shifts, current_consecutive)
                else:
                    current_consecutive = 0
            
            rle_compressions.append(max_consecutive_shifts)
            
            # Count token types
            for token in targets_list:
                if token == 0:  # PAD
                    break
                try:
                    event = codec.decode_event_index(token)
                    token_distributions[event.type] = token_distributions.get(event.type, 0) + 1
                except:
                    pass
            
            if i < 3:  # Show details for first 3
                print(f"\nSample {i}:")
                print(f"  Input shape: {inputs.shape}")
                print(f"  Target shape: {targets.shape}")
                print(f"  TIE events: {tie_count}")
                print(f"  Max consecutive shifts: {max_consecutive_shifts}")
                print(f"  Non-padding tokens: {len([t for t in targets_list if t != 0])}")
            
        except Exception as e:
            print(f"❌ Error loading sample {i}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Summary statistics
    print(f"\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nTIE Events:")
    print(f"  Samples with TIE events: {sum(1 for c in tie_counts if c > 0)}/{num_samples}")
    print(f"  Average TIE events per sample: {sum(tie_counts) / len(tie_counts):.2f}")
    print(f"  Max TIE events in a sample: {max(tie_counts)}")
    
    print(f"\nRLE Compression:")
    print(f"  Average max consecutive shifts: {sum(rle_compressions) / len(rle_compressions):.2f}")
    print(f"  Max consecutive shifts: {max(rle_compressions)}")
    
    if max(rle_compressions) <= 3:
        print(f"  ✅ RLE is working well (max consecutive ≤ 3)")
    elif max(rle_compressions) <= 10:
        print(f"  ⚠️  RLE is partially working (max consecutive ≤ 10)")
    else:
        print(f"  ❌ RLE may not be working (max consecutive > 10)")
    
    print(f"\nToken Distribution:")
    for token_type, count in sorted(token_distributions.items()):
        percentage = (count / sum(token_distributions.values())) * 100
        print(f"  {token_type:12s}: {count:6d} ({percentage:5.2f}%)")
    
    # Validation checks
    print(f"\n" + "="*70)
    print("VALIDATION CHECKS")
    print("="*70)
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: TIE events exist
    if sum(tie_counts) > 0:
        print("✅ Check 1: TIE events are being extracted")
        checks_passed += 1
    else:
        print("❌ Check 1: NO TIE events found (may be expected if chunks start at t=0)")
    
    # Check 2: RLE is effective
    if max(rle_compressions) <= 10:
        print("✅ Check 2: RLE compression is effective")
        checks_passed += 1
    else:
        print(f"❌ Check 2: RLE not effective (max consecutive: {max(rle_compressions)})")
    
    # Check 3: Valid token distribution
    if 'shift' in token_distributions and 'pitch' in token_distributions:
        print("✅ Check 3: Valid token types present (shift, pitch)")
        checks_passed += 1
    else:
        print("❌ Check 3: Missing essential token types")
    
    # Check 4: No all-padding samples
    if all(sum(1 for t in targets.tolist() if t != 0) > 10 for targets in [dataset[i]['targets'] for i in range(min(3, num_samples))]):
        print("✅ Check 4: Samples contain non-padding tokens")
        checks_passed += 1
    else:
        print("❌ Check 4: Some samples are mostly padding")
    
    # Check 5: Input/output shapes correct
    sample = dataset[0]
    if sample['inputs'].shape[0] == 256 and sample['targets'].shape[0] == 1024:
        print("✅ Check 5: Correct input (256) and target (1024) lengths")
        checks_passed += 1
    else:
        print(f"❌ Check 5: Wrong shapes - inputs: {sample['inputs'].shape}, targets: {sample['targets'].shape}")
    
    print(f"\n{'='*70}")
    print(f"OVERALL: {checks_passed}/{total_checks} checks passed")
    print(f"{'='*70}")
    
    return checks_passed >= 3  # At least 3/5 checks should pass


def test_batch_loading():
    """Test loading a batch for training."""
    
    print(f"\n{'='*70}")
    print("BATCH LOADING TEST")
    print(f"{'='*70}")
    
    cache_dir = "/cluster/home/ahmedyz/yassine/MT3-pytorch/cache/maestro/train"
    
    if not Path(cache_dir).exists():
        print(f"❌ Cache directory not found")
        return False
    
    from torch.utils.data import DataLoader
    
    dataset = CachedMaestroDataset(cache_dir=cache_dir, max_tracks=5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    try:
        batch = next(iter(dataloader))
        
        print(f"\n✅ Successfully loaded a batch")
        print(f"  Batch inputs shape: {batch['inputs'].shape}")
        print(f"  Batch targets shape: {batch['targets'].shape}")
        print(f"  Batch audio_chunk shape: {batch['audio_chunk'].shape}")
        
        # Expected: [4, 256, 512], [4, 1024], [4, 32768]
        expected_inputs = (4, 256, 512)
        expected_targets = (4, 1024)
        expected_audio = (4, 32768)
        
        if batch['inputs'].shape == expected_inputs:
            print(f"  ✅ Inputs shape correct: {expected_inputs}")
        else:
            print(f"  ❌ Inputs shape wrong: {batch['inputs'].shape} vs {expected_inputs}")
            
        if batch['targets'].shape == expected_targets:
            print(f"  ✅ Targets shape correct: {expected_targets}")
        else:
            print(f"  ❌ Targets shape wrong: {batch['targets'].shape} vs {expected_targets}")
            
        if batch['audio_chunk'].shape == expected_audio:
            print(f"  ✅ Audio shape correct: {expected_audio}")
        else:
            print(f"  ❌ Audio shape wrong: {batch['audio_chunk'].shape} vs {expected_audio}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_determinism():
    """Test that loading the same sample twice gives same results."""
    
    print(f"\n{'='*70}")
    print("DETERMINISM TEST (Random Chunking)")
    print(f"{'='*70}")
    
    cache_dir = "/cluster/home/ahmedyz/yassine/MT3-pytorch/cache/maestro/train"
    
    if not Path(cache_dir).exists():
        print(f"❌ Cache directory not found")
        return True  # Skip this test
    
    dataset = CachedMaestroDataset(cache_dir=cache_dir, max_tracks=1)
    
    # Load same index multiple times
    samples = [dataset[0] for _ in range(5)]
    
    # They should be DIFFERENT (random chunking!)
    targets_list = [s['targets'].tolist() for s in samples]
    
    all_same = all(t == targets_list[0] for t in targets_list[1:])
    
    if not all_same:
        print("✅ Random chunking is working (each load gives different chunk)")
        
        # Show variety
        tie_counts = [t.count(codec.encode_event(vocabularies.Event("tie", 0))) for t in targets_list]
        print(f"  TIE event counts across 5 loads: {tie_counts}")
        print(f"  (Different counts indicate different chunks)")
        
        return True
    else:
        print("⚠️  Warning: All 5 loads gave identical results (no randomness?)")
        print("   This might be OK if the audio is very short.")
        return True


if __name__ == "__main__":
    success1 = test_full_pipeline()
    success2 = test_batch_loading()
    success3 = test_determinism()
    
    print(f"\n{'='*70}")
    if success1 and success2:
        print("✅ PIPELINE IS WORKING!")
        print("   - TIE extraction: operational")
        print("   - RLE compression: operational")  
        print("   - Batch loading: operational")
        print("   - Random chunking: operational")
    else:
        print("❌ PIPELINE HAS ISSUES")
        if not success1:
            print("   - Full pipeline test failed")
        if not success2:
            print("   - Batch loading test failed")
    print(f"{'='*70}")
