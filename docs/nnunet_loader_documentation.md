# nnUNet3DWeightLoader Documentation

## Overview

The `nnUNet3DWeightLoader` class provides a comprehensive solution for loading 3D UNet weights from nnU-Net format into PyTorch models. This class handles the complex task of mapping between nnU-Net's PlainConvUNet architecture and the project's UNet3D implementation.

## Features

- **Plans.json Parsing**: Automatically loads and parses nnU-Net configuration files
- **Architecture Building**: Constructs 3D UNet models based on nnU-Net specifications
- **Weight Mapping**: Intelligently maps weights between different architecture formats
- **Validation**: Ensures compatibility between loaded weights and target models
- **Preprocessing Config**: Extracts preprocessing parameters for data preparation

## Class Structure

### nnUNet3DWeightLoader

```python
class nnUNet3DWeightLoader:
    def __init__(self, plans_path: str, checkpoint_path: Optional[str] = None)
    def get_architecture_config(self) -> Dict[str, Any]
    def build_nnunet_3d_unet(self, num_classes: int = 1, in_channels: int = 1) -> nn.Module
    def load_nnunet_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, torch.Tensor]
    def map_nnunet_to_unet3d(self, nnunet_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
    def load_weights_to_model(self, model: nn.Module, checkpoint_path: Optional[str] = None, strict: bool = False) -> nn.Module
    def validate_architecture_compatibility(self, model: nn.Module) -> bool
    def get_preprocessing_config(self) -> Dict[str, Any]
    def print_config_summary(self)
```

## Usage Examples

### Basic Usage

```python
from utils.nnunet_loader import nnUNet3DWeightLoader

# Initialize loader
loader = nnUNet3DWeightLoader('plans.json', 'checkpoint.pth')

# Build and load model
model = loader.build_nnunet_3d_unet(num_classes=1)
model = loader.load_weights_to_model(model)

# Print configuration
loader.print_config_summary()
```

### Using Convenience Function

```python
from utils.nnunet_loader import load_nnunet_3d_weights

# One-line loading
model = load_nnunet_3d_weights(
    plans_path='plans.json',
    checkpoint_path='checkpoint.pth',
    num_classes=1
)
```

### Advanced Usage with Custom Model

```python
from models.architectures.unet3d import UNet3D

# Create custom model
model = UNet3D(n_class=2)

# Load weights
loader = nnUNet3DWeightLoader('plans.json')
model = loader.load_weights_to_model(model, 'checkpoint.pth', strict=False)
```

## Configuration Details

### Architecture Configuration

The loader extracts the following parameters from `plans.json`:

```python
{
    'n_stages': 6,                    # Number of encoder/decoder stages
    'features_per_stage': [32, 64, 128, 256, 320, 320],  # Features per stage
    'conv_op': 'torch.nn.modules.conv.Conv3d',           # Convolution operation
    'kernel_sizes': [[3,3,3], ...],  # Kernel sizes for each stage
    'strides': [[1,1,1], [2,2,2], ...],  # Stride patterns
    'n_conv_per_stage': [2, 2, 2, 2, 2, 2],  # Convolutions per stage
    'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d',  # Normalization
    'nonlin': 'torch.nn.LeakyReLU',   # Activation function
    'patch_size': [160, 128, 112],    # Input patch size
    'batch_size': 2                   # Training batch size
}
```

### Preprocessing Configuration

```python
{
    'patch_size': [160, 128, 112],    # Input patch dimensions
    'spacing': [1.0, 1.0, 1.0],       # Voxel spacing
    'normalization_schemes': ['ZScoreNormalization'],  # Normalization method
    'use_mask_for_norm': [True],      # Use mask for normalization
    'median_image_size': [221.0, 214.0, 161.0]  # Median image size
}
```

## Weight Mapping

The class handles complex weight mapping between nnU-Net's PlainConvUNet and the project's UNet3D:

### Encoder Mapping
- `module.encoder.stage0.*` → `down_tr64.*`
- `module.encoder.stage1.*` → `down_tr128.*`
- `module.encoder.stage2.*` → `down_tr256.*`
- `module.encoder.stage3.*` → `down_tr512.*`

### Decoder Mapping
- `module.decoder.stage0.*` → `up_tr64.*`
- `module.decoder.stage1.*` → `up_tr128.*`
- `module.decoder.stage2.*` → `up_tr256.*`

### Layer Type Mapping
- `conv.weight` → `conv1.weight`
- `conv.bias` → `conv1.bias`
- `norm.weight` → `bn1.weight`
- `norm.bias` → `bn1.bias`
- `up.weight` → `up_conv.weight`
- `seg_outputs.*` → `classifier.final_conv.*`

## Error Handling

The class includes comprehensive error handling:

- **File Not Found**: Checks for existence of plans.json and checkpoint files
- **Configuration Validation**: Ensures required keys exist in plans.json
- **Weight Loading**: Handles different checkpoint formats (state_dict, net, direct)
- **Key Mapping**: Gracefully handles missing or mismatched keys
- **Strict Loading**: Supports both strict and non-strict weight loading

## Integration with Existing Code

The loader integrates seamlessly with the existing project structure:

```python
# In fseft/utils.py
from utils.nnunet_loader import nnUNet3DWeightLoader

def load_nnunet_model(plans_path, checkpoint_path, num_classes=1):
    loader = nnUNet3DWeightLoader(plans_path, checkpoint_path)
    model = loader.build_nnunet_3d_unet(num_classes=num_classes)
    model = loader.load_weights_to_model(model)
    return model
```

## Performance Considerations

- **Memory Efficient**: Loads weights directly to CPU to avoid GPU memory issues
- **Lazy Loading**: Only loads weights when explicitly requested
- **Validation**: Minimal overhead validation for compatibility checking
- **Mapping Caching**: Efficient key mapping with minimal string operations

## Troubleshooting

### Common Issues

1. **Key Mismatch**: Use `strict=False` for partial weight loading
2. **Architecture Mismatch**: Verify plans.json matches expected architecture
3. **Memory Issues**: Load weights to CPU first, then move to GPU
4. **Missing Keys**: Check if checkpoint format matches expected structure

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

loader = nnUNet3DWeightLoader('plans.json', 'checkpoint.pth')
# Will show detailed mapping information
```

## Future Enhancements

- Support for additional nnU-Net architectures (ResNet, DenseNet)
- Automatic architecture detection from checkpoint
- Support for multi-GPU weight loading
- Integration with MONAI's nnU-Net implementation
- Support for different normalization schemes

## Dependencies

- PyTorch >= 1.8.0
- JSON (built-in)
- OS (built-in)
- Typing (built-in)

## License

This implementation follows the same license as the parent project.
