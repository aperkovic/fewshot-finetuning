#!/usr/bin/env python3
"""
Checkpoint Inspector Script

This script analyzes PyTorch checkpoint files and extracts detailed information
about model architecture, weights, and structure. The output is logged to a
structured log file that can be used to generate model classes.

Usage:
    python inspect_checkpoint.py --checkpoint_path /path/to/checkpoint.pth
    python inspect_checkpoint.py --checkpoint_path /path/to/checkpoint.pth --output_log model_analysis.log
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn


def setup_logging(log_file: str) -> logging.Logger:
    """Set up logging to both file and console."""
    logger = logging.getLogger('checkpoint_inspector')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def analyze_checkpoint_structure(checkpoint: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Analyze the structure of a checkpoint file."""
    logger.info("=== CHECKPOINT STRUCTURE ANALYSIS ===")
    
    structure_info = {
        'checkpoint_keys': list(checkpoint.keys()),
        'has_state_dict': 'state_dict' in checkpoint,
        'has_net': 'net' in checkpoint,
        'has_model': 'model' in checkpoint,
        'has_optimizer': 'optimizer' in checkpoint,
        'has_scheduler': 'scheduler' in checkpoint,
        'has_epoch': 'epoch' in checkpoint,
        'has_loss': 'loss' in checkpoint,
        'metadata_keys': []
    }
    
    # Log basic structure
    logger.info(f"Checkpoint contains {len(checkpoint)} top-level keys:")
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, dict):
            logger.info(f"  - {key}: dict with {len(value)} items")
        elif isinstance(value, torch.Tensor):
            logger.info(f"  - {key}: tensor with shape {value.shape}")
        else:
            logger.info(f"  - {key}: {type(value).__name__}")
    
    # Extract metadata
    for key in ['epoch', 'best_val_loss', 'best_val_dice', 'lr', 'optimizer_state_dict']:
        if key in checkpoint:
            structure_info['metadata_keys'].append(key)
            logger.info(f"Metadata - {key}: {checkpoint[key]}")
    
    return structure_info


def analyze_state_dict(state_dict: Dict[str, torch.Tensor], logger: logging.Logger) -> Dict[str, Any]:
    """Analyze the state dictionary to understand model architecture."""
    logger.info("=== STATE DICT ANALYSIS ===")
    
    analysis = {
        'total_parameters': 0,
        'total_layers': len(state_dict),
        'layer_info': {},
        'architecture_hints': [],
        'input_channels': None,
        'output_channels': None,
        'feature_dimensions': [],
        'conv_layers': [],
        'linear_layers': [],
        'norm_layers': [],
        'activation_layers': [],
        'pooling_layers': [],
        'upsampling_layers': []
    }
    
    # Analyze each layer
    for name, tensor in state_dict.items():
        layer_info = {
            'name': name,
            'shape': list(tensor.shape),
            'num_params': tensor.numel(),
            'dtype': str(tensor.dtype),
            'requires_grad': tensor.requires_grad if hasattr(tensor, 'requires_grad') else 'unknown'
        }
        
        analysis['layer_info'][name] = layer_info
        analysis['total_parameters'] += tensor.numel()
        
        # Categorize layers
        if 'conv' in name.lower():
            analysis['conv_layers'].append(name)
            if 'weight' in name and len(tensor.shape) >= 4:
                # Conv layer - analyze input/output channels
                if len(tensor.shape) == 4:  # 2D conv
                    in_channels, out_channels = tensor.shape[0], tensor.shape[1]
                elif len(tensor.shape) == 5:  # 3D conv
                    in_channels, out_channels = tensor.shape[0], tensor.shape[1]
                else:
                    in_channels, out_channels = None, None
                
                if in_channels is not None:
                    analysis['feature_dimensions'].append((in_channels, out_channels))
                    
        elif 'linear' in name.lower() or 'fc' in name.lower():
            analysis['linear_layers'].append(name)
        elif 'norm' in name.lower() or 'bn' in name.lower():
            analysis['norm_layers'].append(name)
        elif 'relu' in name.lower() or 'leaky' in name.lower() or 'gelu' in name.lower():
            analysis['activation_layers'].append(name)
        elif 'pool' in name.lower():
            analysis['pooling_layers'].append(name)
        elif 'up' in name.lower() or 'transpose' in name.lower():
            analysis['upsampling_layers'].append(name)
    
    # Detect architecture patterns
    if len(analysis['conv_layers']) > 10:
        analysis['architecture_hints'].append("Deep CNN architecture")
    if len(analysis['linear_layers']) > 0:
        analysis['architecture_hints'].append("Contains fully connected layers")
    if any('3d' in name.lower() for name in analysis['conv_layers']):
        analysis['architecture_hints'].append("3D convolutional architecture")
    if any('swin' in name.lower() for name in state_dict.keys()):
        analysis['architecture_hints'].append("Swin Transformer architecture")
    if any('unet' in name.lower() for name in state_dict.keys()):
        analysis['architecture_hints'].append("UNet-like architecture")
    
    # Try to infer input/output channels
    if analysis['feature_dimensions']:
        # First conv layer typically shows input channels
        first_conv = analysis['feature_dimensions'][0]
        analysis['input_channels'] = first_conv[0]
        
        # Last conv layer typically shows output channels
        last_conv = analysis['feature_dimensions'][-1]
        analysis['output_channels'] = last_conv[1]
    
    # Log summary
    logger.info(f"Total parameters: {analysis['total_parameters']:,}")
    logger.info(f"Total layers: {analysis['total_layers']}")
    logger.info(f"Convolutional layers: {len(analysis['conv_layers'])}")
    logger.info(f"Linear layers: {len(analysis['linear_layers'])}")
    logger.info(f"Normalization layers: {len(analysis['norm_layers'])}")
    
    if analysis['input_channels']:
        logger.info(f"Inferred input channels: {analysis['input_channels']}")
    if analysis['output_channels']:
        logger.info(f"Inferred output channels: {analysis['output_channels']}")
    
    logger.info(f"Architecture hints: {', '.join(analysis['architecture_hints'])}")
    
    return analysis


def analyze_tensor_shapes(state_dict: Dict[str, torch.Tensor], logger: logging.Logger) -> Dict[str, Any]:
    """Analyze tensor shapes to understand model dimensions."""
    logger.info("=== TENSOR SHAPE ANALYSIS ===")
    
    shape_analysis = {
        'unique_shapes': {},
        'conv_kernels': [],
        'feature_maps': [],
        'weight_matrices': []
    }
    
    for name, tensor in state_dict.items():
        shape = tuple(tensor.shape)
        if shape not in shape_analysis['unique_shapes']:
            shape_analysis['unique_shapes'][shape] = []
        shape_analysis['unique_shapes'][shape].append(name)
        
        # Categorize by shape patterns
        if len(shape) == 5 and 'conv' in name.lower():  # 3D conv kernel
            shape_analysis['conv_kernels'].append((name, shape))
        elif len(shape) == 4 and 'conv' in name.lower():  # 2D conv kernel
            shape_analysis['conv_kernels'].append((name, shape))
        elif len(shape) == 2:  # Linear layer weight matrix
            shape_analysis['weight_matrices'].append((name, shape))
        elif len(shape) == 1:  # Bias or normalization parameters
            shape_analysis['feature_maps'].append((name, shape))
    
    # Log shape analysis
    logger.info("Unique tensor shapes found:")
    for shape, layers in shape_analysis['unique_shapes'].items():
        logger.info(f"  Shape {shape}: {len(layers)} layers")
        if len(layers) <= 5:  # Only show layer names if not too many
            for layer in layers:
                logger.info(f"    - {layer}")
    
    return shape_analysis


def generate_model_requirements(analysis: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Generate requirements for creating a model class."""
    logger.info("=== MODEL CLASS REQUIREMENTS ===")
    
    requirements = {
        'expected_input_shape': None,
        'expected_output_shape': None,
        'required_parameters': {},
        'architecture_type': 'unknown',
        'forward_method_signature': 'forward(self, x)',
        'initialization_parameters': {},
        'key_components': []
    }
    
    # Determine architecture type
    if 'Swin Transformer architecture' in analysis['architecture_hints']:
        requirements['architecture_type'] = 'swin_transformer'
        requirements['key_components'].extend(['swinViT', 'classifier'])
    elif 'UNet-like architecture' in analysis['architecture_hints']:
        requirements['architecture_type'] = 'unet'
        requirements['key_components'].extend(['encoder', 'decoder', 'bottleneck'])
    elif 'Deep CNN architecture' in analysis['architecture_hints']:
        requirements['architecture_type'] = 'cnn'
        requirements['key_components'].extend(['conv_layers', 'pooling', 'classifier'])
    
    # Infer input/output requirements
    if analysis['input_channels']:
        requirements['initialization_parameters']['in_channels'] = analysis['input_channels']
    if analysis['output_channels']:
        requirements['initialization_parameters']['out_channels'] = analysis['output_channels']
    
    # Add common parameters
    requirements['initialization_parameters']['num_classes'] = analysis['output_channels'] or 1
    
    # Log requirements
    logger.info(f"Architecture type: {requirements['architecture_type']}")
    logger.info(f"Key components: {requirements['key_components']}")
    logger.info(f"Initialization parameters: {requirements['initialization_parameters']}")
    
    return requirements


def save_analysis_to_file(analysis_data: Dict[str, Any], output_file: str, logger: logging.Logger):
    """Save the complete analysis to a JSON file."""
    logger.info(f"=== SAVING ANALYSIS TO {output_file} ===")
    
    # Convert any non-serializable objects
    serializable_data = {}
    for key, value in analysis_data.items():
        if isinstance(value, dict):
            serializable_data[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    serializable_data[key][sub_key] = {
                        'shape': list(sub_value.shape),
                        'dtype': str(sub_value.dtype),
                        'numel': sub_value.numel()
                    }
                else:
                    serializable_data[key][sub_key] = sub_value
        else:
            serializable_data[key] = value
    
    with open(output_file, 'w') as f:
        json.dump(serializable_data, f, indent=2, default=str)
    
    logger.info(f"Analysis saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Inspect PyTorch checkpoint files')
    parser.add_argument('--checkpoint_path', required=True, help='Path to checkpoint file')
    parser.add_argument('--output_log', default='checkpoint_analysis.log', help='Output log file')
    parser.add_argument('--output_json', default='checkpoint_analysis.json', help='Output JSON analysis file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_log)
    logger.info(f"Starting checkpoint analysis of: {args.checkpoint_path}")
    
    # Check if file exists
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    
    try:
        # Load checkpoint
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
        logger.info("Checkpoint loaded successfully")
        
        # Analyze checkpoint structure
        structure_info = analyze_checkpoint_structure(checkpoint, logger)
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'net' in checkpoint:
            state_dict = checkpoint['net']
        else:
            state_dict = checkpoint
        
        # Analyze state dict
        state_analysis = analyze_state_dict(state_dict, logger)
        
        # Analyze tensor shapes
        shape_analysis = analyze_tensor_shapes(state_dict, logger)
        
        # Generate model requirements
        model_requirements = generate_model_requirements(state_analysis, logger)
        
        # Combine all analysis
        complete_analysis = {
            'checkpoint_path': args.checkpoint_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'structure_info': structure_info,
            'state_analysis': state_analysis,
            'shape_analysis': shape_analysis,
            'model_requirements': model_requirements
        }
        
        # Save analysis
        save_analysis_to_file(complete_analysis, args.output_json, logger)
        
        logger.info("Checkpoint analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
