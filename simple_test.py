#!/usr/bin/env python3
"""
Simple test to debug the nnU-Net checkpoint structure.
"""

import torch
import os
from collections import OrderedDict

def debug_checkpoint_structure(checkpoint_path):
    """Debug the structure of the nnU-Net checkpoint."""
    
    print("="*60)
    print("nnU-Net Checkpoint Structure Debug")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Load with compatibility mode
        print("\n1. Loading checkpoint...")
        import torch.serialization
        torch.serialization.add_safe_globals([
            'numpy.core.multiarray.scalar',
            'numpy.dtype', 
            'numpy.ndarray',
            'numpy.core.multiarray._reconstruct'
        ])
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✓ Checkpoint loaded successfully!")
        
        # Debug the structure
        print(f"\n2. Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"   Top-level keys: {list(checkpoint.keys())}")
            
            # Look for state dict
            state_dict = None
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("   Found 'state_dict' key")
            elif 'net' in checkpoint:
                state_dict = checkpoint['net']
                print("   Found 'net' key")
            else:
                state_dict = checkpoint
                print("   Using checkpoint as state dict")
            
            print(f"   State dict type: {type(state_dict)}")
            
            if hasattr(state_dict, 'keys'):
                print(f"   State dict keys: {list(state_dict.keys())}")
                
                # Check the structure of each key
                tensor_count = 0
                for key, value in state_dict.items():
                    print(f"\n   Key: {key}")
                    print(f"   Type: {type(value)}")
                    
                    if hasattr(value, 'shape'):
                        print(f"   Shape: {value.shape}")
                        tensor_count += 1
                    elif isinstance(value, (dict, OrderedDict)):
                        print(f"   Sub-keys: {list(value.keys())}")
                        # Check sub-values
                        for sub_key, sub_value in value.items():
                            print(f"     {sub_key}: {type(sub_value)}", end="")
                            if hasattr(sub_value, 'shape'):
                                print(f" {sub_value.shape}")
                                tensor_count += 1
                            else:
                                print()
                    else:
                        print(f"   Value: {value}")
                
                print(f"\n   Total tensors found: {tensor_count}")
                
                # Try to extract all tensors
                print("\n3. Extracting all tensors...")
                all_tensors = {}
                
                def extract_tensors(obj, prefix=""):
                    """Recursively extract tensors from nested structure."""
                    if hasattr(obj, 'shape'):  # This is a tensor
                        return {prefix: obj}
                    elif hasattr(obj, 'keys'):  # This is a dict/OrderedDict
                        result = {}
                        for key, value in obj.items():
                            new_prefix = f"{prefix}.{key}" if prefix else key
                            result.update(extract_tensors(value, new_prefix))
                        return result
                    else:
                        return {}
                
                all_tensors = extract_tensors(state_dict)
                print(f"   Extracted {len(all_tensors)} tensors")
                
                # Show some examples
                print("\n   Sample tensors:")
                for i, (key, tensor) in enumerate(list(all_tensors.items())[:5]):
                    print(f"     {key}: {tensor.shape}")
                
                if len(all_tensors) > 5:
                    print(f"     ... and {len(all_tensors) - 5} more")
                
                return True
            else:
                print("   State dict has no keys() method")
                return False
        else:
            print("   Checkpoint is not a dictionary")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    checkpoint_path = "/home/sagemaker-user/fewshot-finetuning/models/pretrained_weights/nnunet_ALLALLDATA/fold_0/checkpoint_final.pth"
    
    success = debug_checkpoint_structure(checkpoint_path)
    
    if success:
        print("\n" + "="*60)
        print("✓ Checkpoint structure analysis complete!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Checkpoint analysis failed!")
        print("="*60)

if __name__ == "__main__":
    main()
