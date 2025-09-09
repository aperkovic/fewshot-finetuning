#!/usr/bin/env python3
"""
Example usage of nnUNet3DWeightLoader for loading 3D UNet weights from nnU-Net.

This script demonstrates how to:
1. Load nnU-Net plans.json configuration
2. Build a 3D UNet model based on nnU-Net specifications
3. Load weights from nnU-Net checkpoint
4. Use the model for inference or fine-tuning
"""

import sys
import os
import torch
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nnunet_loader import nnUNet3DWeightLoader, load_nnunet_3d_weights, load_nnunet_checkpoint_safe
from models.architectures.unet3d import UNet3D


def main():
    parser = argparse.ArgumentParser(description='nnU-Net 3D Weight Loading Example')
    parser.add_argument('--plans_path', type=str, default='./plans.json',
                       help='Path to nnU-Net plans.json file')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to nnU-Net checkpoint file')
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of output classes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    
    args = parser.parse_args()
    
    print("="*60)
    print("nnU-Net 3D Weight Loading Example")
    print("="*60)
    
    # Method 1: Using the convenience function
    print("\n1. Loading model using convenience function...")
    try:
        model = load_nnunet_3d_weights(
            plans_path=args.plans_path,
            checkpoint_path=args.checkpoint_path,
            num_classes=args.num_classes
        )
        print("✓ Model loaded successfully using convenience function")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Method 2: Using the class directly for more control
    print("\n2. Loading model using class directly...")
    try:
        # Initialize loader
        loader = nnUNet3DWeightLoader(args.plans_path, args.checkpoint_path)
        
        # Print configuration summary
        loader.print_config_summary()
        
        # Get architecture configuration
        arch_config = loader.get_architecture_config()
        print(f"\nArchitecture configuration:")
        print(f"  - Stages: {arch_config['n_stages']}")
        print(f"  - Features per stage: {arch_config['features_per_stage']}")
        print(f"  - Patch size: {arch_config['patch_size']}")
        
        # Get preprocessing configuration
        preprocess_config = loader.get_preprocessing_config()
        print(f"\nPreprocessing configuration:")
        print(f"  - Normalization: {preprocess_config['normalization_schemes']}")
        print(f"  - Spacing: {preprocess_config['spacing']}")
        
        # Build model
        model2 = loader.build_nnunet_3d_unet(num_classes=args.num_classes)
        
        # Load weights
        model2 = loader.load_weights_to_model(model2, strict=False)
        
        # Validate compatibility
        is_compatible = loader.validate_architecture_compatibility(model2)
        print(f"\nArchitecture compatibility: {'✓ Compatible' if is_compatible else '✗ Incompatible'}")
        
        print("✓ Model loaded successfully using class directly")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Test safe checkpoint loading
    print("\n3. Testing safe checkpoint loading...")
    try:
        # Test the safe loading function directly
        state_dict = load_nnunet_checkpoint_safe(args.checkpoint_path)
        print(f"✓ Safe loading successful - loaded {len(state_dict)} parameters")
        
        # Show some example keys
        print("Sample parameter keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            print(f"  {key}: {state_dict[key].shape}")
        if len(state_dict) > 5:
            print(f"  ... and {len(state_dict) - 5} more parameters")
            
    except Exception as e:
        print(f"✗ Error during safe loading: {e}")
        return

    # Test model inference
    print("\n4. Testing model inference...")
    try:
        model.eval()
        model = model.to(args.device)
        
        # Create dummy input based on patch size from plans
        patch_size = loader.config_3d['patch_size']
        dummy_input = torch.randn(1, 1, *patch_size).to(args.device)
        
        print(f"Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print("✓ Model inference successful")
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return
    
    # Model statistics
    print("\n5. Model statistics...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


def demonstrate_weight_mapping():
    """Demonstrate the weight mapping functionality."""
    print("\n" + "="*40)
    print("Weight Mapping Demonstration")
    print("="*40)
    
    # Create a dummy nnU-Net state dict
    dummy_nnunet_state = {
        'module.encoder.stage0.conv.weight': torch.randn(32, 1, 3, 3, 3),
        'module.encoder.stage0.conv.bias': torch.randn(32),
        'module.encoder.stage0.norm.weight': torch.randn(32),
        'module.encoder.stage0.norm.bias': torch.randn(32),
        'module.decoder.stage0.up.weight': torch.randn(16, 32, 2, 2, 2),
        'module.decoder.stage0.up.bias': torch.randn(16),
        'module.seg_outputs.weight': torch.randn(1, 32, 1, 1, 1),
        'module.seg_outputs.bias': torch.randn(1),
    }
    
    # Create loader and test mapping
    loader = nnUNet3DWeightLoader('./plans.json')
    mapped_state = loader.map_nnunet_to_unet3d(dummy_nnunet_state)
    
    print("Original nnU-Net keys:")
    for key in dummy_nnunet_state.keys():
        print(f"  {key}")
    
    print("\nMapped UNet3D keys:")
    for key in mapped_state.keys():
        print(f"  {key}")
    
    print(f"\nMapped {len(dummy_nnunet_state)} keys to {len(mapped_state)} keys")


if __name__ == "__main__":
    # Run the main example
    main()
    
    # Demonstrate weight mapping
    demonstrate_weight_mapping()
