"""
SynthSeg Model Architecture for FSEFT Pipeline
Integrates SynthSeg models with the FSEFT fine-tuning framework
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model

# SynthSeg imports (these would need to be installed)
try:
    from ext.lab2im import utils as lab2im_utils
    from ext.lab2im import layers as lab2im_layers
    from ext.lab2im import edit_volumes
    from ext.neuron import models as nrn_models
    SYNTHSEG_AVAILABLE = True
except ImportError:
    SYNTHSEG_AVAILABLE = False
    print("Warning: SynthSeg dependencies not available. Please install SynthSeg first.")


class SynthSegWrapper(nn.Module):
    """
    Wrapper for SynthSeg models to integrate with FSEFT pipeline.
    This maintains the original SynthSeg architecture while making it compatible
    with the FSEFT Linear Probe and PEFT methods.
    """
    
    def __init__(self, synthseg_model, feature_size: int = 24, n_labels: int = 29):
        super().__init__()
        self.synthseg_model = synthseg_model
        self.feature_size = feature_size
        self.n_labels = n_labels
        
        # Store original segmentation layers for potential restoration
        self.original_seg_layers = None
        
    def forward(self, x):
        """
        Forward pass through SynthSeg model.
        Returns the feature maps from the decoder (before final segmentation layers).
        """
        # Convert PyTorch tensor to numpy for SynthSeg
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
            
        # SynthSeg expects specific input format
        # Add batch dimension if needed
        if len(x_np.shape) == 4:  # (C, D, H, W)
            x_np = np.expand_dims(x_np, 0)  # Add batch dimension
        
        # Get predictions from SynthSeg
        predictions = self.synthseg_model.predict(x_np)
        
        # Convert back to PyTorch tensor
        if isinstance(predictions, np.ndarray):
            predictions = torch.from_numpy(predictions).float()
        
        # For FSEFT, we want the feature maps before the final segmentation
        # SynthSeg returns the final segmentation, so we need to extract features
        # This is a simplified approach - in practice, you might need to modify
        # the SynthSeg model to expose intermediate features
        
        # For now, we'll use the predictions as features (this is not ideal)
        # In a real implementation, you'd want to modify SynthSeg to expose
        # intermediate feature maps
        feature_maps = predictions
        
        return feature_maps
    
    def get_feature_size(self):
        """Get the feature size for the final feature maps."""
        return self.feature_size
    
    def get_output_channels(self):
        """Get the number of output channels."""
        return self.n_labels


def build_synthseg_model(
    path_model: str, 
    input_shape: Tuple[int, ...],
    labels_segmentation: np.ndarray,
    n_levels: int = 5,
    nb_conv_per_level: int = 2,
    conv_size: int = 3,
    unet_feat_count: int = 24,
    feat_multiplier: int = 2,
    activation: str = 'elu',
    sigma_smoothing: float = 0.5,
    flip_indices: Optional[np.ndarray] = None,
    gradients: bool = False) -> Model:
    """
    Build SynthSeg model based on the original predict.py implementation.
    
    Args:
        path_model: Path to the trained SynthSeg model
        input_shape: Input shape for the model
        labels_segmentation: List of segmentation labels
        n_levels: Number of levels for UNet
        nb_conv_per_level: Number of convolution layers per level
        conv_size: Size of UNet's convolution masks
        unet_feat_count: Number of features for the first layer
        feat_multiplier: Multiplicative factor for features
        activation: Activation function
        sigma_smoothing: Standard deviation for Gaussian smoothing
        flip_indices: Indices for flipping augmentation
        gradients: Whether to use gradient input
        
    Returns:
        Built SynthSeg model
    """
    if not SYNTHSEG_AVAILABLE:
        raise ImportError("SynthSeg dependencies not available. Please install SynthSeg first.")
    
    assert os.path.isfile(path_model), "The provided model path does not exist."
    
    # Get labels
    n_labels_seg = len(labels_segmentation)
    
    if gradients:
        input_image = KL.Input(input_shape)
        last_tensor = lab2im_layers.ImageGradients('sobel', True)(input_image)
        last_tensor = KL.Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x) + K.epsilon()))(last_tensor)
        net = Model(inputs=input_image, outputs=last_tensor)
    else:
        net = None
    
    # Build UNet
    net = nrn_models.unet(
        input_model=net,
        input_shape=input_shape,
        nb_labels=n_labels_seg,
        nb_levels=n_levels,
        nb_conv_per_level=nb_conv_per_level,
        conv_size=conv_size,
        nb_features=unet_feat_count,
        feat_mult=feat_multiplier,
        activation=activation,
        batch_norm=-1)
    net.load_weights(path_model, by_name=True)
    
    # Smooth posteriors if specified
    if sigma_smoothing > 0:
        last_tensor = net.output
        last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
        last_tensor = lab2im_layers.GaussianBlur(sigma=sigma_smoothing)(last_tensor)
        net = Model(inputs=net.inputs, outputs=last_tensor)
    
    if flip_indices is not None:
        # Segment flipped image
        input_image = net.inputs[0]
        seg = net.output
        image_flipped = lab2im_layers.RandomFlip(axis=0, prob=1)(input_image)
        last_tensor = net(image_flipped)
        
        # Flip back and re-order channels
        last_tensor = lab2im_layers.RandomFlip(axis=0, prob=1)(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.split(x, [1] * n_labels_seg, axis=-1), name='split')(last_tensor)
        reordered_channels = [last_tensor[flip_indices[i]] for i in range(n_labels_seg)]
        last_tensor = KL.Lambda(lambda x: tf.concat(x, -1), name='concat')(reordered_channels)
        
        # Average two segmentations and build model
        name_segm_prediction_layer = 'average_lr'
        last_tensor = KL.Lambda(lambda x: 0.5 * (x[0] + x[1]), name=name_segm_prediction_layer)([seg, last_tensor])
        net = Model(inputs=net.inputs, outputs=last_tensor)
    
    return net


def load_synthseg_model(model_path: str, 
                       labels_segmentation: np.ndarray,
                       input_shape: Tuple[int, ...] = (None, None, None, 1),
                       n_levels: int = 5,
                       nb_conv_per_level: int = 2,
                       conv_size: int = 3,
                       unet_feat_count: int = 24,
                       feat_multiplier: int = 2,
                       activation: str = 'elu',
                       sigma_smoothing: float = 0.5,
                       flip_indices: Optional[np.ndarray] = None,
                       gradients: bool = False,
                       device: str = 'cpu') -> SynthSegWrapper:
    """
    Load SynthSeg model and wrap it for FSEFT compatibility.
    
    Args:
        model_path: Path to the SynthSeg model file
        labels_segmentation: List of segmentation labels
        input_shape: Input shape for the model
        n_levels: Number of UNet levels
        nb_conv_per_level: Number of convolutions per level
        conv_size: Convolution kernel size
        unet_feat_count: Number of features in first layer
        feat_multiplier: Feature multiplier
        activation: Activation function
        sigma_smoothing: Gaussian smoothing sigma
        flip_indices: Indices for flipping augmentation
        gradients: Whether to use gradient input
        device: Device to load model on
        
    Returns:
        SynthSegWrapper instance
    """
    if not SYNTHSEG_AVAILABLE:
        raise ImportError("SynthSeg dependencies not available. Please install SynthSeg first.")
    
    # Build the SynthSeg model
    synthseg_model = build_synthseg_model(
        path_model=model_path,
        input_shape=input_shape,
        labels_segmentation=labels_segmentation,
        n_levels=n_levels,
        nb_conv_per_level=nb_conv_per_level,
        conv_size=conv_size,
        unet_feat_count=unet_feat_count,
        feat_multiplier=feat_multiplier,
        activation=activation,
        sigma_smoothing=sigma_smoothing,
        flip_indices=flip_indices,
        gradients=gradients
    )
    
    # Create wrapper
    wrapper = SynthSegWrapper(
        synthseg_model=synthseg_model,
        feature_size=unet_feat_count,
        n_labels=len(labels_segmentation)
    )
    
    return wrapper


def get_synthseg_config(model_path: str, 
                       labels_segmentation: np.ndarray) -> Dict[str, Any]:
    """
    Get configuration for SynthSeg model.
    
    Args:
        model_path: Path to the SynthSeg model
        labels_segmentation: List of segmentation labels
        
    Returns:
        Configuration dictionary
    """
    return {
        "architecture": "synthseg",
        "model_path": model_path,
        "n_labels": len(labels_segmentation),
        "feature_size": 24,  # Default SynthSeg feature size
        "input_shape": (None, None, None, 1),  # 3D + channels
        "n_levels": 5,
        "nb_conv_per_level": 2,
        "conv_size": 3,
        "unet_feat_count": 24,
        "feat_multiplier": 2,
        "activation": "elu",
        "sigma_smoothing": 0.5,
        "gradients": False,
        "flip": True,
        "target_res": 1.0,
        "roi_x": 160,
        "roi_y": 160, 
        "roi_z": 160,
        "space_x": 1.0,
        "space_y": 1.0,
        "space_z": 1.0,
        "channelOut": len(labels_segmentation)
    }
