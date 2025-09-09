#!/usr/bin/env python3
"""
Test the updated nnUNet3DWeightLoader with the actual checkpoint structure.
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.nnunet_loader import nnUNet3DWeightLoader

def test_updated_loader():
    """Test the updated loader with the actual checkpoint."""
    
    checkpoint_path = "/home/sagemaker-user/fewshot-finetuning/models/pretrained_weights/nnunet_ALLALLDATA/fold_0/checkpoint_final.pth"
    plans_path = "plans.json"
    
    print("="*60)
    print("Testing Updated nnUNet3DWeightLoader")
    print("="*60)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    if not os.path.exists(plans_path):
        print(f"❌ Plans not found: {plans_path}")
        return False
    
    try:
        # Initialize loader
        print("\n1. Initializing loader...")
        loader = nnUNet3DWeightLoader(plans_path, checkpoint_path)
        print("✓ Loader initialized")
        
        # Print configuration
        print("\n2. Configuration summary:")
        loader.print_config_summary()
        
        # Load checkpoint
        print("\n3. Loading checkpoint...")
        nnunet_state = loader.load_nnunet_checkpoint()
        print(f"✓ Loaded {len(nnunet_state)} parameters from nnU-Net checkpoint")
        
        # Test weight mapping
        print("\n4. Testing weight mapping...")
        mapped_dict = loader.map_nnunet_to_unet3d(nnunet_state)
        print(f"✓ Mapped to {len(mapped_dict)} UNet3D parameters")
        
        # Show some mapping examples
        print("\n5. Mapping examples:")
        sample_keys = list(nnunet_state.keys())[:10]
        for orig_key in sample_keys:
            mapped_key = None
            for mapped_k in mapped_dict.keys():
                if any(part in mapped_k for part in orig_key.split('.')[-2:]):
                    mapped_key = mapped_k
                    break
            print(f"  {orig_key[:50]:<50} -> {mapped_key}")
        
        # Build and test model
        print("\n6. Building and testing model...")
        from models.architectures.unet3d import UNet3D
        
        model = loader.build_nnunet_3d_unet(num_classes=1)
        print(f"✓ Built model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Load weights
        model = loader.load_weights_to_model(model, strict=False)
        print("✓ Loaded weights into model")
        
        # Test inference
        print("\n7. Testing inference...")
        model.eval()
        patch_size = loader.config_3d['patch_size']
        dummy_input = torch.randn(1, 1, *patch_size)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Inference successful: {dummy_input.shape} -> {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        print("\n" + "="*60)
        print("🎉 All tests passed! nnU-Net weight loading is working!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_updated_loader()
    
    if success:
        print("\n✅ You can now use the nnUNet3DWeightLoader class successfully!")
        print("\nExample usage:")
        print("```python")
        print("from utils.nnunet_loader import load_nnunet_3d_weights")
        print("model = load_nnunet_3d_weights('plans.json', 'checkpoint_final.pth')")
        print("```")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
