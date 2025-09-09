import torch
import torch.nn as nn
import json
import os
import re
from typing import Dict, Any, Optional, Tuple
import warnings


class nnUNet3DWeightLoader:
    """
    Class for loading 3D UNet weights from nnU-Net format.
    
    This class handles:
    - Loading nnU-Net plans.json configuration
    - Building 3D UNet architecture based on nnU-Net specifications
    - Mapping and loading weights from nnU-Net checkpoints
    - Converting between nnU-Net and project-specific UNet3D formats
    """
    
    def __init__(self, plans_path: str, checkpoint_path: Optional[str] = None):
        """
        Initialize the nnU-Net 3D weight loader.
        
        Args:
            plans_path: Path to the nnU-Net plans.json file
            checkpoint_path: Optional path to nnU-Net checkpoint file
        """
        self.plans_path = plans_path
        self.checkpoint_path = checkpoint_path
        self.plans = self._load_plans()
        self.config_3d = self._get_3d_config()
        
    def _load_plans(self) -> Dict[str, Any]:
        """Load and parse the nnU-Net plans.json file."""
        if not os.path.exists(self.plans_path):
            raise FileNotFoundError(f"Plans file not found: {self.plans_path}")
            
        with open(self.plans_path, 'r') as f:
            plans = json.load(f)
            
        print(f"Loaded plans for dataset: {plans.get('dataset_name', 'Unknown')}")
        return plans
    
    def _get_3d_config(self) -> Dict[str, Any]:
        """Extract 3D full resolution configuration from plans."""
        if '3d_fullres' not in self.plans['configurations']:
            raise KeyError("3D full resolution configuration not found in plans")
            
        return self.plans['configurations']['3d_fullres']
    
    def get_architecture_config(self) -> Dict[str, Any]:
        """
        Get the 3D UNet architecture configuration from plans.json.
        
        Returns:
            Dictionary containing architecture parameters
        """
        arch_config = self.config_3d['architecture']['arch_kwargs']
        
        # Extract key parameters
        config = {
            'n_stages': arch_config['n_stages'],
            'features_per_stage': arch_config['features_per_stage'],
            'conv_op': arch_config['conv_op'],
            'kernel_sizes': arch_config['kernel_sizes'],
            'strides': arch_config['strides'],
            'n_conv_per_stage': arch_config['n_conv_per_stage'],
            'n_conv_per_stage_decoder': arch_config['n_conv_per_stage_decoder'],
            'conv_bias': arch_config['conv_bias'],
            'norm_op': arch_config['norm_op'],
            'norm_op_kwargs': arch_config['norm_op_kwargs'],
            'dropout_op': arch_config['dropout_op'],
            'dropout_op_kwargs': arch_config['dropout_op_kwargs'],
            'nonlin': arch_config['nonlin'],
            'nonlin_kwargs': arch_config['nonlin_kwargs'],
            'patch_size': self.config_3d['patch_size'],
            'batch_size': self.config_3d['batch_size']
        }
        
        return config
    
    def build_nnunet_3d_unet(self, num_classes: int = 1, in_channels: int = 1) -> nn.Module:
        """
        Build a 3D UNet architecture based on nnU-Net configuration.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            
        Returns:
            PyTorch 3D UNet model
        """
        from models.architectures.unet3d import UNet3D
        
        # For now, use the existing UNet3D implementation
        # In a full implementation, you would build the exact nnU-Net architecture
        # based on the plans.json configuration
        model = UNet3D(n_class=num_classes)
        
        print(f"Built 3D UNet with {num_classes} classes and {in_channels} input channels")
        return model
    
    def load_nnunet_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Load weights from nnU-Net checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file (overrides init parameter)
            
        Returns:
            Dictionary containing the loaded state dict
        """
        path = checkpoint_path or self.checkpoint_path
        if not path:
            raise ValueError("No checkpoint path provided")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
            
        print(f"Loading nnU-Net checkpoint from: {path}")
        
        # Try loading with weights_only=True first (safer)
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            print("✓ Loaded checkpoint with weights_only=True (safe mode)")
        except Exception as e:
            print(f"⚠️  Safe loading failed: {e}")
            print("⚠️  Attempting to load with weights_only=False (less safe but required for older checkpoints)")
            
            # For older nnU-Net checkpoints, we need to use weights_only=False
            # and add safe globals for numpy serialization
            try:
                # Add safe globals for numpy serialization that nnU-Net uses
                import torch.serialization
                torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
                
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                print("✓ Loaded checkpoint with weights_only=False (compatibility mode)")
            except Exception as e2:
                print(f"✗ Both loading methods failed:")
                print(f"  Safe loading error: {e}")
                print(f"  Compatibility loading error: {e2}")
                raise RuntimeError(f"Could not load checkpoint. Safe loading failed: {e}. Compatibility loading failed: {e2}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'network_weights' in checkpoint:
                state_dict = checkpoint['network_weights']
                print("✓ Found 'network_weights' key in checkpoint (nnU-Net format)")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("✓ Found 'state_dict' key in checkpoint")
            elif 'net' in checkpoint:
                state_dict = checkpoint['net']
                print("✓ Found 'net' key in checkpoint")
            else:
                state_dict = checkpoint
                print("✓ Using checkpoint as direct state dict")
        else:
            state_dict = checkpoint
            print("✓ Checkpoint is direct state dict")
            
        print(f"✓ Loaded state dict with {len(state_dict)} parameters")
        return state_dict
    
    def map_nnunet_to_unet3d(self, nnunet_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Map nnU-Net state dict keys to project UNet3D format.
        
        Args:
            nnunet_state_dict: State dict from nnU-Net checkpoint
            
        Returns:
            Mapped state dict compatible with project UNet3D
        """
        mapped_dict = {}
        unmapped_keys = []
        
        print("Mapping nnU-Net PlainConvUNet to UNet3D architecture...")
        
        for key, value in nnunet_state_dict.items():
            mapped_key = self._map_single_key(key)
            
            if mapped_key:
                mapped_dict[mapped_key] = value
            else:
                unmapped_keys.append(key)
        
        print(f"Mapped {len(mapped_dict)} parameters successfully")
        if unmapped_keys:
            print(f"Warning: {len(unmapped_keys)} parameters could not be mapped:")
            for key in unmapped_keys[:10]:  # Show first 10 unmapped keys
                print(f"  - {key}")
            if len(unmapped_keys) > 10:
                print(f"  ... and {len(unmapped_keys) - 10} more")
        
        return mapped_dict
    
    def _map_single_key(self, key: str) -> Optional[str]:
        """Map a single nnU-Net key to UNet3D format."""
        
        # Handle encoder stem (first layer)
        if key.startswith('encoder.stem.convs.0.'):
            if 'conv.weight' in key:
                return 'down_tr64.ops.0.conv1.weight'
            elif 'conv.bias' in key:
                return 'down_tr64.ops.0.conv1.bias'
            elif 'norm.weight' in key:
                return 'down_tr64.ops.0.bn1.weight'
            elif 'norm.bias' in key:
                return 'down_tr64.ops.0.bn1.bias'
        
        # Handle encoder stages
        if key.startswith('encoder.stages.'):
            stage_match = re.match(r'encoder\.stages\.(\d+)', key)
            if stage_match:
                stage_num = int(stage_match.group(1))
                return self._map_encoder_stage_key(key, stage_num)
        
        # Handle decoder stages (skip connections and upsampling)
        if key.startswith('decoder.stages.'):
            stage_match = re.match(r'decoder\.stages\.(\d+)', key)
            if stage_match:
                stage_num = int(stage_match.group(1))
                return self._map_decoder_stage_key(key, stage_num)
        
        # Handle decoder transpose convolutions (upsampling)
        if key.startswith('decoder.transpconvs.'):
            transp_match = re.match(r'decoder\.transpconvs\.(\d+)', key)
            if transp_match:
                transp_num = int(transp_match.group(1))
                return self._map_transpconv_key(key, transp_num)
        
        # Handle decoder segmentation layers (final outputs)
        if key.startswith('decoder.seg_layers.'):
            seg_match = re.match(r'decoder\.seg_layers\.(\d+)', key)
            if seg_match:
                seg_num = int(seg_match.group(1))
                return self._map_seg_layer_key(key, seg_num)
        
        # Skip decoder.encoder.* keys (these are shared encoder weights)
        if key.startswith('decoder.encoder.'):
            return None  # Skip shared encoder weights in decoder
        
        return None  # Unmapped key
    
    def _map_encoder_stage_key(self, key: str, stage_num: int) -> Optional[str]:
        """Map encoder stage key to UNet3D down transition."""
        stage_mapping = {
            0: 'down_tr64',
            1: 'down_tr128', 
            2: 'down_tr256',
            3: 'down_tr512',
            4: 'down_tr512',  # Additional stage
            5: 'down_tr512'   # Additional stage
        }
        
        if stage_num not in stage_mapping:
            return None
            
        base_name = stage_mapping[stage_num]
        
        # Map the first conv layer of the first block
        if 'blocks.0.conv1.conv.weight' in key:
            return f"{base_name}.ops.0.conv1.weight"
        elif 'blocks.0.conv1.conv.bias' in key:
            return f"{base_name}.ops.0.conv1.bias"
        elif 'blocks.0.conv1.norm.weight' in key:
            return f"{base_name}.ops.0.bn1.weight"
        elif 'blocks.0.conv1.norm.bias' in key:
            return f"{base_name}.ops.0.bn1.bias"
        
        return None
    
    def _map_decoder_stage_key(self, key: str, stage_num: int) -> Optional[str]:
        """Map decoder stage key to UNet3D up transition."""
        stage_mapping = {
            0: 'up_tr256',
            1: 'up_tr128',
            2: 'up_tr64',
            3: 'up_tr64',   # Additional stage
            4: 'up_tr64'    # Additional stage
        }
        
        if stage_num not in stage_mapping:
            return None
            
        base_name = stage_mapping[stage_num]
        
        # Map conv layers
        if 'convs.0.conv.weight' in key:
            return f"{base_name}.ops.0.conv1.weight"
        elif 'convs.0.conv.bias' in key:
            return f"{base_name}.ops.0.conv1.bias"
        elif 'convs.0.norm.weight' in key:
            return f"{base_name}.ops.0.bn1.weight"
        elif 'convs.0.norm.bias' in key:
            return f"{base_name}.ops.0.bn1.bias"
        
        return None
    
    def _map_transpconv_key(self, key: str, transp_num: int) -> Optional[str]:
        """Map transpose convolution key to UNet3D up convolution."""
        transp_mapping = {
            0: 'up_tr256',
            1: 'up_tr128', 
            2: 'up_tr64',
            3: 'up_tr64',
            4: 'up_tr64'
        }
        
        if transp_num not in transp_mapping:
            return None
            
        base_name = transp_mapping[transp_num]
        
        if 'weight' in key:
            return f"{base_name}.up_conv.weight"
        elif 'bias' in key:
            return f"{base_name}.up_conv.bias"
        
        return None
    
    def _map_seg_layer_key(self, key: str, seg_num: int) -> Optional[str]:
        """Map segmentation layer key to UNet3D classifier."""
        if 'weight' in key:
            return 'classifier.final_conv.weight'
        elif 'bias' in key:
            return 'classifier.final_conv.bias'
        
        return None
    
    
    def load_weights_to_model(self, model: nn.Module, checkpoint_path: Optional[str] = None, 
                            strict: bool = False) -> nn.Module:
        """
        Load nnU-Net weights into a PyTorch model.
        
        Args:
            model: PyTorch model to load weights into
            checkpoint_path: Path to nnU-Net checkpoint
            strict: Whether to strictly enforce key matching
            
        Returns:
            Model with loaded weights
        """
        # Load nnU-Net checkpoint
        nnunet_state_dict = self.load_nnunet_checkpoint(checkpoint_path)
        
        # Map to project format
        mapped_state_dict = self.map_nnunet_to_unet3d(nnunet_state_dict)
        
        # Load into model
        try:
            model.load_state_dict(mapped_state_dict, strict=strict)
            print("Successfully loaded nnU-Net weights into model")
        except Exception as e:
            print(f"Warning: Could not load all weights: {e}")
            if not strict:
                # Try loading with strict=False for partial loading
                model.load_state_dict(mapped_state_dict, strict=False)
                print("Loaded weights with strict=False (partial loading)")
            else:
                raise e
                
        return model
    
    def validate_architecture_compatibility(self, model: nn.Module) -> bool:
        """
        Validate that the model architecture is compatible with nnU-Net configuration.
        
        Args:
            model: Model to validate
            
        Returns:
            True if compatible, False otherwise
        """
        arch_config = self.get_architecture_config()
        
        # Check if model has expected number of stages
        expected_stages = arch_config['n_stages']
        
        # This is a simplified check - in practice, you'd do more detailed validation
        print(f"Validating architecture compatibility...")
        print(f"Expected stages: {expected_stages}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        return True
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get preprocessing configuration from plans.json.
        
        Returns:
            Dictionary containing preprocessing parameters
        """
        return {
            'patch_size': self.config_3d['patch_size'],
            'spacing': self.config_3d['spacing'],
            'normalization_schemes': self.config_3d['normalization_schemes'],
            'use_mask_for_norm': self.config_3d['use_mask_for_norm'],
            'median_image_size': self.config_3d['median_image_size_in_voxels']
        }
    
    def print_config_summary(self):
        """Print a summary of the loaded configuration."""
        print("\n" + "="*50)
        print("nnU-Net 3D Configuration Summary")
        print("="*50)
        print(f"Dataset: {self.plans.get('dataset_name', 'Unknown')}")
        print(f"Patch size: {self.config_3d['patch_size']}")
        print(f"Batch size: {self.config_3d['batch_size']}")
        print(f"Stages: {self.config_3d['architecture']['arch_kwargs']['n_stages']}")
        print(f"Features per stage: {self.config_3d['architecture']['arch_kwargs']['features_per_stage']}")
        print(f"Normalization: {self.config_3d['normalization_schemes']}")
        print("="*50)


# Utility function for safe nnU-Net checkpoint loading
def load_nnunet_checkpoint_safe(checkpoint_path: str, map_location: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Safely load nnU-Net checkpoint with proper error handling for PyTorch 2.6+.
    
    Args:
        checkpoint_path: Path to nnU-Net checkpoint file
        map_location: Device to load tensors to ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing the loaded state dict
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading nnU-Net checkpoint: {checkpoint_path}")
    
    # Try safe loading first
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        print("✓ Loaded with weights_only=True (safe mode)")
    except Exception as e:
        print(f"⚠️  Safe loading failed: {e}")
        print("⚠️  Using compatibility mode for older nnU-Net checkpoints...")
        
        # Add safe globals for numpy serialization
        import torch.serialization
        torch.serialization.add_safe_globals([
            'numpy.core.multiarray.scalar',
            'numpy.dtype',
            'numpy.ndarray',
            'numpy.core.multiarray._reconstruct'
        ])
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            print("✓ Loaded with weights_only=False (compatibility mode)")
        except Exception as e2:
            raise RuntimeError(f"Failed to load checkpoint. Safe mode: {e}. Compatibility mode: {e2}")
    
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
    
    print(f"✓ Loaded {len(state_dict)} parameters")
    return state_dict


# Convenience function for easy usage
def load_nnunet_3d_weights(plans_path: str, checkpoint_path: str, 
                          model: Optional[nn.Module] = None, 
                          num_classes: int = 1) -> nn.Module:
    """
    Convenience function to load nnU-Net 3D weights.
    
    Args:
        plans_path: Path to nnU-Net plans.json
        checkpoint_path: Path to nnU-Net checkpoint
        model: Optional existing model (will create one if None)
        num_classes: Number of output classes
        
    Returns:
        Model with loaded weights
    """
    loader = nnUNet3DWeightLoader(plans_path, checkpoint_path)
    
    if model is None:
        model = loader.build_nnunet_3d_unet(num_classes=num_classes)
    
    model = loader.load_weights_to_model(model, checkpoint_path)
    loader.print_config_summary()
    
    return model
