#!/usr/bin/env python3
"""
Test script for SynthSeg integration with FSEFT pipeline
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


def test_synthseg_config():
    """Test SynthSeg configuration loading"""
    print("Testing SynthSeg configuration...")
    
    # Create mock args
    class MockArgs:
        def __init__(self):
            self.model_id = "synthseg"
    
    args = MockArgs()
    get_model_config(args)
    
    # Check if SynthSeg config is loaded
    assert args.model_cfg["architecture"] == "synthseg"
    assert args.model_cfg["channelOut"] == 29
    assert args.model_cfg["fout"] == 24
    assert args.model_cfg["roi_x"] == 160
    assert args.model_cfg["roi_y"] == 160
    assert args.model_cfg["roi_z"] == 160
    
    print("✓ SynthSeg configuration loaded successfully")
    return args


def test_synthseg_model_loading():
    """Test SynthSeg model loading (without actual model file)"""
    print("Testing SynthSeg model loading...")
    
    # Test configuration
    args = test_synthseg_config()
    
    # Add required attributes for model loading
    args.method = "scratch"
    args.decoder = False
    args.bottleneck = False
    args.out_channels = 29
    args.universal_indexes = None
    args.objective = "multiclass"
    
    try:
        # This will fail if SynthSeg dependencies are not installed
        model = load_model(args)
        print("✓ SynthSeg model loaded successfully")
        return model
    except ImportError as e:
        print(f"⚠ SynthSeg dependencies not available: {e}")
        print("  Please install SynthSeg dependencies:")
        print("  pip install tensorflow>=2.8.0 keras>=2.8.0 neuron lab2im")
        return None
    except Exception as e:
        print(f"✗ Error loading SynthSeg model: {e}")
        return None


def test_synthseg_model_validation():
    """Test SynthSeg model file validation"""
    print("Testing SynthSeg model validation...")
    
    # Test with non-existent file
    fake_path = "./models/pretrained_weights/synthseg_fake.h5"
    assert not validate_synthseg_model(fake_path)
    print("✓ Correctly identified non-existent model file")
    
    # Test with existing file (if it exists)
    real_path = "./models/pretrained_weights/synthseg.h5"
    if os.path.exists(real_path):
        assert validate_synthseg_model(real_path)
        print("✓ Correctly identified existing model file")
    else:
        print("⚠ SynthSeg model file not found (expected for testing)")
    
    print("✓ SynthSeg model validation working correctly")


def test_synthseg_architecture():
    """Test SynthSeg architecture wrapper"""
    print("Testing SynthSeg architecture wrapper...")
    
    try:
        from models.architectures.synthseg import SynthSegWrapper, get_synthseg_config
        
        # Test configuration generation
        labels = np.arange(29)
        config = get_synthseg_config("./test_model.h5", labels)
        
        assert config["architecture"] == "synthseg"
        assert config["n_labels"] == 29
        assert config["feature_size"] == 24
        
        print("✓ SynthSeg architecture wrapper working correctly")
        return True
        
    except ImportError as e:
        print(f"⚠ SynthSeg architecture not available: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing SynthSeg architecture: {e}")
        return False


def test_synthseg_integration():
    """Test full SynthSeg integration with FSEFT pipeline"""
    print("Testing SynthSeg integration with FSEFT pipeline...")
    
    # Test configuration
    args = test_synthseg_config()
    
    # Add required attributes
    args.method = "lora"  # Test with PEFT method
    args.decoder = False
    args.bottleneck = False
    args.out_channels = 29
    args.universal_indexes = None
    args.objective = "multiclass"
    args.adapt_hp = {}  # Will be set by set_model_peft
    
    try:
        # Load model
        model = load_model(args)
        if model is None:
            print("⚠ Skipping integration test due to missing dependencies")
            return False
        
        # Test PEFT setup
        set_model_peft(model, args)
        
        print("✓ SynthSeg integration with FSEFT pipeline working correctly")
        return True
        
    except ImportError as e:
        print(f"⚠ SynthSeg integration test skipped due to missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"✗ Error in SynthSeg integration test: {e}")
        return False


def test_synthseg_prediction():
    """Test SynthSeg prediction functionality"""
    print("Testing SynthSeg prediction functionality...")
    
    try:
        from models.architectures.synthseg import SynthSegWrapper
        
        # Create mock model wrapper
        class MockSynthSegModel:
            def predict(self, x):
                # Mock prediction that returns random output
                return np.random.rand(*x.shape)
        
        mock_model = MockSynthSegModel()
        wrapper = SynthSegWrapper(mock_model, feature_size=24, n_labels=29)
        
        # Test forward pass
        input_tensor = torch.randn(1, 1, 32, 32, 32)  # Batch, Channel, D, H, W
        output = wrapper.forward(input_tensor)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == input_tensor.shape[0]  # Same batch size
        
        print("✓ SynthSeg prediction functionality working correctly")
        return True
        
    except ImportError as e:
        print(f"⚠ SynthSeg prediction test skipped due to missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"✗ Error in SynthSeg prediction test: {e}")
        return False


def main():
    """Run all SynthSeg integration tests"""
    print("=" * 60)
    print("SynthSeg Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_synthseg_config),
        ("Model Validation", test_synthseg_model_validation),
        ("Architecture Wrapper", test_synthseg_architecture),
        ("Model Loading", test_synthseg_model_loading),
        ("Prediction Functionality", test_synthseg_prediction),
        ("FSEFT Integration", test_synthseg_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! SynthSeg integration is working correctly.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        print("\nTo complete the integration:")
        print("1. Install SynthSeg dependencies: pip install tensorflow>=2.8.0 keras>=2.8.0 neuron lab2im")
        print("2. Download a SynthSeg model and place it at ./models/pretrained_weights/synthseg.h5")
        print("3. Run the tests again")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
