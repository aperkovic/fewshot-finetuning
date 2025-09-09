#!/usr/bin/env python3
"""
Helper script to inspect nnU-Net model structure and help with conversion.

This script will:
1. Load your nnU-Net model
2. Print the layer structure
3. Help identify the correct mapping for the framework
4. Save a converted model if needed

Usage:
    python inspect_nnunet_model.py --model_path /path/to/your/nnunet/model.pth
"""

import argparse
import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.architectures.unet3d import UNet3D


def inspect_model_structure(model_path):
    """Inspect the structure of the nnU-Net model."""
    print("=" * 60)
    print("nnU-Net Model Inspection")
    print("=" * 60)
    
    # Load the model
    print(f"Loading model from: {model_path}")
    model_dict = torch.load(model_path, map_location='cpu')
    
    # Identify the state dict
    if 'state_dict' in model_dict:
        state_dict = model_dict["state_dict"]
        print("Found 'state_dict' key in model")
    elif 'net' in model_dict:
        state_dict = model_dict["net"]
        print("Found 'net' key in model")
    elif 'model_state_dict' in model_dict:
        state_dict = model_dict["model_state_dict"]
        print("Found 'model_state_dict' key in model")
    else:
        state_dict = model_dict
        print("Using raw model dictionary")
    
    print(f"\nTotal layers in nnU-Net model: {len(state_dict)}")
    print("\nFirst 10 layers:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {i+1:2d}. {key} -> {state_dict[key].shape}")
    
    print("\nLast 10 layers:")
    for i, key in enumerate(list(state_dict.keys())[-10:]):
        print(f"  {len(state_dict)-9+i:2d}. {key} -> {state_dict[key].shape}")
    
    # Analyze layer patterns
    print("\n" + "=" * 60)
    print("Layer Pattern Analysis")
    print("=" * 60)
    
    patterns = {}
    for key in state_dict.keys():
        # Extract the main component (before the first dot)
        component = key.split('.')[0] if '.' in key else key
        if component not in patterns:
            patterns[component] = []
        patterns[component].append(key)
    
    print("Layer components found:")
    for component, layers in patterns.items():
        print(f"  {component}: {len(layers)} layers")
        if len(layers) <= 5:  # Show all if few layers
            for layer in layers:
                print(f"    - {layer}")
        else:  # Show first few if many layers
            for layer in layers[:3]:
                print(f"    - {layer}")
            print(f"    ... and {len(layers)-3} more")
    
    return state_dict


def compare_with_framework(state_dict):
    """Compare nnU-Net structure with framework UNet3D."""
    print("\n" + "=" * 60)
    print("Framework Compatibility Analysis")
    print("=" * 60)
    
    # Create a framework model to get its structure
    framework_model = UNet3D(n_class=1)
    framework_state = framework_model.state_dict()
    
    print(f"Framework UNet3D layers: {len(framework_state)}")
    print("\nFramework layer structure:")
    for key in framework_state.keys():
        print(f"  - {key} -> {framework_state[key].shape}")
    
    # Try to find matches
    print("\n" + "=" * 60)
    print("Potential Mappings")
    print("=" * 60)
    
    # Common mappings
    mappings = {
        'conv_blocks_context': 'down_tr',
        'conv_blocks_localization': 'up_tr',
        'seg_outputs': 'classifier',
        'final': 'classifier'
    }
    
    for nnunet_pattern, framework_pattern in mappings.items():
        matching_layers = [k for k in state_dict.keys() if nnunet_pattern in k]
        if matching_layers:
            print(f"\n{nnunet_pattern} -> {framework_pattern}:")
            for layer in matching_layers[:5]:  # Show first 5
                print(f"  {layer}")
            if len(matching_layers) > 5:
                print(f"  ... and {len(matching_layers)-5} more")


def suggest_conversion_mapping(state_dict):
    """Suggest a conversion mapping based on the model structure."""
    print("\n" + "=" * 60)
    print("Suggested Conversion Mapping")
    print("=" * 60)
    
    # This is a template - you'll need to adjust based on your specific model
    print("Based on the analysis, here's a suggested mapping:")
    print("(You may need to adjust these based on your specific nnU-Net model)")
    print()
    
    # Analyze the structure and suggest mappings
    encoder_layers = [k for k in state_dict.keys() if 'conv_blocks_context' in k]
    decoder_layers = [k for k in state_dict.keys() if 'conv_blocks_localization' in k]
    output_layers = [k for k in state_dict.keys() if any(x in k for x in ['seg_outputs', 'final', 'output'])]
    
    print("Encoder layers found:")
    for layer in encoder_layers[:5]:
        print(f"  {layer}")
    if len(encoder_layers) > 5:
        print(f"  ... and {len(encoder_layers)-5} more")
    
    print("\nDecoder layers found:")
    for layer in decoder_layers[:5]:
        print(f"  {layer}")
    if len(decoder_layers) > 5:
        print(f"  ... and {len(decoder_layers)-5} more")
    
    print("\nOutput layers found:")
    for layer in output_layers:
        print(f"  {layer}")
    
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("1. Review the layer mappings above")
    print("2. Update the load_nnunet_weights function in utils/models.py")
    print("3. Adjust the mapping based on your specific model structure")
    print("4. Test the conversion with your model")


def main():
    parser = argparse.ArgumentParser(description='Inspect nnU-Net model structure')
    parser.add_argument('--model_path', required=True, 
                       help='Path to your nnU-Net model weights')
    parser.add_argument('--save_analysis', default=None,
                       help='Save analysis to file (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    try:
        # Inspect the model
        state_dict = inspect_model_structure(args.model_path)
        
        # Compare with framework
        compare_with_framework(state_dict)
        
        # Suggest mappings
        suggest_conversion_mapping(state_dict)
        
        if args.save_analysis:
            print(f"\nSaving analysis to: {args.save_analysis}")
            # You could save the analysis to a file here
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
