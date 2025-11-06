import torch
import sys

def inspect_file(path):
    print(f"--- Inspecting file: {path} ---")
    try:
        data = torch.load(path, map_location='cpu')
        print("--- File loaded successfully ---")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("\n--- Data Content Inspection ---")
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"\nKey: '{key}'")
            if isinstance(value, torch.Tensor):
                print(f"  Type: torch.Tensor")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
            elif isinstance(value, list):
                print(f"  Type: list")
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  First element type: {type(value[0])}")
                    if isinstance(value[0], torch.Tensor):
                         print(f"  First element shape: {value[0].shape}")
                         print(f"  First element dtype: {value[0].dtype}")
            else:
                print(f"  Type: {type(value)}")
    else:
        print(f"Data type: {type(data)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_pt_file.py <path_to_pt_file>")
    else:
        inspect_file(sys.argv[1])
