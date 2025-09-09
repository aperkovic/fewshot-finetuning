#!/usr/bin/env python3
"""
Test script for loading nnU-Net checkpoint with PyTorch 2.6+ compatibility.
This script specifically handles the error you encountered.
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.nnunet_loader import load_nnunet_checkpoint_safe, nnUNet3DWeightLoader


def test_checkpoint_loading(checkpoint_path: str, plans_path: str = "plans.json"):
    """
    Test loading the specific nnU-Net checkpoint that was causing issues.
    """
    print("="*60)
    print("nnU-Net Checkpoint Loading Test")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Plans: {plans_path}")
    print("="*60)
    
    # Test 1: Direct safe loading
    print("\n1. Testing direct safe loading...")
    try:
        state_dict = load_nnunet_checkpoint_safe(checkpoint_path)
        print(f"✓ Successfully loaded {len(state_dict)} parameters")
        
        # Show first few keys
        print("\nFirst 10 parameter keys:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {i+1:2d}. {key}: {state_dict[key].shape}")
        
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more parameters")
            
    except Exception as e:
        print(f"✗ Direct loading failed: {e}")
        return False
    
    # Test 2: Using the full loader class
    print("\n2. Testing with nnUNet3DWeightLoader...")
    try:
        loader = nnUNet3DWeightLoader(plans_path, checkpoint_path)
        
        # Print configuration
        loader.print_config_summary()
        
        # Load checkpoint through the class
        nnunet_state = loader.load_nnunet_checkpoint()
        print(f"✓ Loaded through class: {len(nnunet_state)} parameters")
        
    except Exception as e:
        print(f"✗ Class loading failed: {e}")
        return False
    
    # Test 3: Test weight mapping
    print("\n3. Testing weight mapping...")
    try:
        mapped_dict = loader.map_nnunet_to_unet3d(nnunet_state)
        print(f"✓ Mapped to UNet3D format: {len(mapped_dict)} parameters")
        
        # Show mapping examples
        print("\nMapping examples:")
        original_keys = list(nnunet_state.keys())[:5]
        for orig_key in original_keys:
            mapped_key = None
            for mapped_k in mapped_dict.keys():
                if any(part in mapped_k for part in orig_key.split('.')[-2:]):
                    mapped_key = mapped_k
                    break
            print(f"  {orig_key} -> {mapped_key}")
        
    except Exception as e:
        print(f"✗ Weight mapping failed: {e}")
        return False
    
    # Test 4: Build and load model
    print("\n4. Testing model building and loading...")
    try:
        from models.architectures.unet3d import UNet3D
        
        # Build model
        model = loader.build_nnunet_3d_unet(num_classes=1)
        print(f"✓ Built model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Load weights
        model = loader.load_weights_to_model(model, strict=False)
        print("✓ Successfully loaded weights into model")
        
        # Test inference
        model.eval()
        patch_size = loader.config_3d['patch_size']
        dummy_input = torch.randn(1, 1, *patch_size)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Model inference successful: {dummy_input.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"✗ Model building/loading failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All tests passed! nnU-Net checkpoint loading is working correctly.")
    print("="*60)
    return True


def main():
    # Your specific checkpoint path
    checkpoint_path = "/home/sagemaker-user/fewshot-finetuning/models/pretrained_weights/nnunet_ALLALLDATA/fold_0/checkpoint_final.pth"
    plans_path = "plans.json"
    
    # Check if files exist
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable with the correct path.")
        return
    
    if not os.path.exists(plans_path):
        print(f"❌ Plans file not found: {plans_path}")
        print("Please ensure plans.json is in the current directory.")
        return
    
    # Run tests
    success = test_checkpoint_loading(checkpoint_path, plans_path)
    
    if success:
        print("\n🎉 Success! You can now use the nnUNet3DWeightLoader class to load your nnU-Net weights.")
        print("\nExample usage:")
        print("```python")
        print("from utils.nnunet_loader import load_nnunet_3d_weights")
        print("model = load_nnunet_3d_weights('plans.json', 'checkpoint_final.pth')")
        print("```")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
