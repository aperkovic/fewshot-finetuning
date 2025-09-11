#!/usr/bin/env python3
"""
Example usage of SynthSeg integration with FSEFT pipeline
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.configs import get_model_config
from fseft.utils import load_model, set_model_peft
from utils.synthseg_loader import create_synthseg_model, validate_synthseg_model


def create_mock_args():
    """Create mock arguments for testing"""
    class MockArgs:
        def __init__(self):
            self.model_id = "synthseg"
            self.method = "lora"  # or "scratch", "ft", etc.
            self.decoder = False
            self.bottleneck = False
            self.out_channels = 29
            self.universal_indexes = None
            self.objective = "multiclass"
            self.adapt_hp = {}
    
    return MockArgs()


def example_basic_usage():
    """Example of basic SynthSeg usage"""
    print("=== Basic SynthSeg Usage ===")
    
    # Create arguments
    args = create_mock_args()
    
    # Load model configuration
    get_model_config(args)
    print(f"Model architecture: {args.model_cfg['architecture']}")
    print(f"Feature size: {args.model_cfg['fout']}")
    print(f"Output channels: {args.model_cfg['channelOut']}")
    
    # Load model
    try:
        model = load_model(args)
        print("✓ SynthSeg model loaded successfully")
        
        # Test forward pass
        input_tensor = torch.randn(1, 1, 160, 160, 160)  # Batch, Channel, D, H, W
        with torch.no_grad():
            output = model(input_tensor)
            print(f"Input shape: {input_tensor.shape}")
            print(f"Output shape: {output.shape}")
        
        return model
    except ImportError as e:
        print(f"⚠ SynthSeg dependencies not available: {e}")
        print("Please install: pip install tensorflow>=2.8.0 keras>=2.8.0 neuron lab2im")
        return None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def example_peft_usage():
    """Example of SynthSeg with PEFT methods"""
    print("\n=== SynthSeg with PEFT Methods ===")
    
    # Create arguments for different PEFT methods
    peft_methods = ["lora", "adapter", "bias", "affine"]
    
    for method in peft_methods:
        print(f"\n--- Testing {method.upper()} ---")
        
        args = create_mock_args()
        args.method = method
        
        try:
            # Load model
            model = load_model(args)
            if model is None:
                continue
            
            # Set up PEFT
            set_model_peft(model, args)
            print(f"✓ {method.upper()} PEFT setup completed")
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable ratio: {trainable_params/total_params:.2%}")
            
        except Exception as e:
            print(f"✗ Error with {method}: {e}")


def example_custom_configuration():
    """Example of custom SynthSeg configuration"""
    print("\n=== Custom SynthSeg Configuration ===")
    
    # Create custom configuration
    class CustomArgs:
        def __init__(self):
            self.model_id = "synthseg"
            self.method = "scratch"
            self.decoder = False
            self.bottleneck = False
            self.out_channels = 15  # Custom number of output channels
            self.universal_indexes = None
            self.objective = "multiclass"
            self.adapt_hp = {}
    
    args = CustomArgs()
    
    # Load base configuration
    get_model_config(args)
    
    # Modify configuration
    args.model_cfg["channelOut"] = 15
    args.model_cfg["roi_x"] = 128
    args.model_cfg["roi_y"] = 128
    args.model_cfg["roi_z"] = 128
    args.model_cfg["fout"] = 32
    
    print(f"Custom configuration:")
    print(f"  Output channels: {args.model_cfg['channelOut']}")
    print(f"  ROI size: {args.model_cfg['roi_x']}x{args.model_cfg['roi_y']}x{args.model_cfg['roi_z']}")
    print(f"  Feature size: {args.model_cfg['fout']}")
    
    try:
        model = load_model(args)
        print("✓ Custom SynthSeg model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading custom model: {e}")


def example_model_validation():
    """Example of SynthSeg model validation"""
    print("\n=== SynthSeg Model Validation ===")
    
    # Test different model paths
    model_paths = [
        "./models/pretrained_weights/synthseg.h5",
        "./models/pretrained_weights/synthseg_fake.h5",
        "./models/pretrained_weights/",
    ]
    
    for path in model_paths:
        exists = os.path.exists(path)
        valid = validate_synthseg_model(path)
        print(f"Path: {path}")
        print(f"  Exists: {exists}")
        print(f"  Valid: {valid}")
        print()


def example_inference_pipeline():
    """Example of complete inference pipeline"""
    print("\n=== Complete Inference Pipeline ===")
    
    args = create_mock_args()
    
    try:
        # Load model
        model = load_model(args)
        if model is None:
            print("⚠ Skipping inference example due to missing dependencies")
            return
        
        # Set up PEFT
        set_model_peft(model, args)
        
        # Create sample data
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 160, 160, 160)
        
        print(f"Input shape: {input_tensor.shape}")
        
        # Inference
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            print(f"Output shape: {output.shape}")
            
            # Convert to probabilities (if needed)
            if output.shape[1] > 1:  # Multi-class
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                print(f"Predictions shape: {predictions.shape}")
                print(f"Unique predictions: {torch.unique(predictions)}")
        
        print("✓ Inference pipeline completed successfully")
        
    except Exception as e:
        print(f"✗ Error in inference pipeline: {e}")


def main():
    """Run all examples"""
    print("SynthSeg Integration Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_peft_usage()
    example_custom_configuration()
    example_model_validation()
    example_inference_pipeline()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use SynthSeg in your own code:")
    print("1. Set model_id='synthseg' in your arguments")
    print("2. Call get_model_config(args) to load configuration")
    print("3. Call load_model(args) to load the model")
    print("4. Call set_model_peft(model, args) for PEFT methods")
    print("5. Use the model for inference or training")


if __name__ == "__main__":
    main()
