#!/usr/bin/env python3
"""
Quick test to load your specific nnU-Net checkpoint.
Run this to test the fix for the PyTorch 2.6+ loading issue.
"""

import torch
import os

def test_nnunet_loading():
    """Test loading the specific checkpoint that was failing."""
    
    checkpoint_path = "/home/sagemaker-user/fewshot-finetuning/models/pretrained_weights/nnunet_ALLALLDATA/fold_0/checkpoint_final.pth"
    
    print("Testing nnU-Net checkpoint loading...")
    print(f"Checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Method 1: Try safe loading first
        print("\n1. Trying safe loading (weights_only=True)...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            print("✓ Safe loading successful!")
            state_dict = checkpoint
        except Exception as e:
            print(f"⚠️  Safe loading failed: {e}")
            print("\n2. Trying compatibility loading (weights_only=False)...")
            
            # Add safe globals for numpy serialization
            import torch.serialization
            torch.serialization.add_safe_globals([
                'numpy.core.multiarray.scalar',
                'numpy.dtype', 
                'numpy.ndarray',
                'numpy.core.multiarray._reconstruct'
            ])
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            print("✓ Compatibility loading successful!")
            
            # Extract state dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'net' in checkpoint:
                    state_dict = checkpoint['net']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
        
        # Convert OrderedDict to regular dict if needed and extract actual tensors
        if hasattr(state_dict, 'keys'):
            # Check if this is a nested structure
            actual_state_dict = {}
            for key, value in state_dict.items():
                if hasattr(value, 'shape'):  # This is a tensor
                    actual_state_dict[key] = value
                elif hasattr(value, 'keys'):  # This is another dict/OrderedDict
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, 'shape'):  # This is a tensor
                            actual_state_dict[f"{key}.{sub_key}"] = sub_value
                        else:
                            print(f"  Warning: Skipping non-tensor {key}.{sub_key}: {type(sub_value)}")
                else:
                    print(f"  Warning: Skipping non-tensor {key}: {type(value)}")
            state_dict = actual_state_dict
        
        print(f"\n✓ Successfully loaded {len(state_dict)} parameters!")
        
        # Show some example keys
        print("\nSample parameter keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            if hasattr(state_dict[key], 'shape'):
                print(f"  {key}: {state_dict[key].shape}")
            else:
                print(f"  {key}: {type(state_dict[key])}")
        
        if len(state_dict) > 5:
            print(f"  ... and {len(state_dict) - 5} more parameters")
        
        print("\n🎉 Checkpoint loading is now working!")
        return True
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        return False

if __name__ == "__main__":
    success = test_nnunet_loading()
    
    if success:
        print("\n" + "="*50)
        print("SUCCESS! You can now use the nnUNet3DWeightLoader class.")
        print("="*50)
        print("\nExample usage:")
        print("```python")
        print("from utils.nnunet_loader import load_nnunet_3d_weights")
        print("model = load_nnunet_3d_weights('plans.json', 'checkpoint_final.pth')")
        print("```")
    else:
        print("\n" + "="*50)
        print("FAILED! Please check the error messages above.")
        print("="*50)
