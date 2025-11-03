"""
Test the main training script with A100 configs.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def test_training_script_help():
    """Test that training script shows correct help."""
    print("=" * 80)
    print("Testing training script --help")
    print("=" * 80)
    
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "training" / "train_multitrack.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode != 0:
        print(f"❌ FAILED: Script returned {result.returncode}")
        print(result.stderr)
        return False
    
    # Check that new config names are present
    required_configs = [
        "A100_pretraining_full",
        "A100_pretraining_test",
        "A100_finetuning_full",
        "A100_finetuning_test",
    ]
    
    for config_name in required_configs:
        if config_name not in result.stdout:
            print(f"❌ FAILED: Config '{config_name}' not found in help")
            print(result.stdout)
            return False
        print(f"✓ Found config: {config_name}")
    
    print("✅ Training script help looks correct!")
    return True


def test_config_import():
    """Test that configs can be imported."""
    print("\n" + "=" * 80)
    print("Testing config import")
    print("=" * 80)
    
    try:
        from config.training_config import get_config
        
        for config_name in [
            "A100_pretraining_full",
            "A100_pretraining_test",
            "A100_finetuning_full",
            "A100_finetuning_test",
        ]:
            config = get_config(config_name)
            
            # Check validation cache dirs are present
            if "maestro_val_cache_dir" not in config:
                print(f"❌ FAILED: 'maestro_val_cache_dir' missing in {config_name}")
                return False
            if "slakh_val_cache_dir" not in config:
                print(f"❌ FAILED: 'slakh_val_cache_dir' missing in {config_name}")
                return False
            
            print(f"✓ Config '{config_name}' has validation cache dirs")
        
        print("✅ All configs have validation cache directories!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TRAINING SCRIPT TEST")
    print("=" * 80)
    
    success = True
    
    if not test_training_script_help():
        success = False
    
    if not test_config_import():
        success = False
    
    if success:
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED")
        print("=" * 80)
        print("\nReady to run training with:")
        print("  python training/train_multitrack.py --config A100_pretraining_test")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
