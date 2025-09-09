#!/usr/bin/env python3
"""
Example script showing how to use nnU-Net trained models with the FSEFT framework.

This script demonstrates how to:
1. Load your nnU-Net trained 3D UNet model
2. Use it with the few-shot fine-tuning framework
3. Run adaptation experiments

Usage:
    python example_nnunet_usage.py --nnunet_model_path /path/to/your/nnunet/model.pth
"""

import argparse
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_fseft import process


def main():
    parser = argparse.ArgumentParser(description='Example usage of nnU-Net with FSEFT framework')
    
    # Required: Path to your nnU-Net model
    parser.add_argument('--nnunet_model_path', required=True, 
                       help='Path to your nnU-Net trained model weights (.pth file)')
    
    # Framework arguments
    parser.add_argument('--out_path', default='./local_data/results/')
    parser.add_argument('--model_id', default='nnunet', 
                       help='Use nnunet model configuration')
    parser.add_argument('--dataset', default="totalseg", 
                       help='Dataset to use for fine-tuning')
    parser.add_argument('--organ', default='liver',
                       help='Target organ for segmentation')
    parser.add_argument('--k', default=1, type=int, 
                       help='Number of shots for few-shot learning')
    parser.add_argument('--seeds', default=3, type=int, 
                       help='Number of experimental seeds')
    
    # Fine-tuning method
    parser.add_argument('--method', default='LP',
                       help='Fine-tuning method: LP, FT, LoRA, etc.')
    parser.add_argument('--decoder', default="frozen", 
                       help='Decoder setting: frozen, fine-tuned, new')
    parser.add_argument('--bottleneck', default="frozen", 
                       help='Bottleneck setting: frozen, fine-tuned, new')
    
    # Training parameters
    parser.add_argument('--early_stop_criteria', default='train', 
                       help='Early stopping criteria: train, val, none')
    parser.add_argument('--num_workers', default=0, type=int, 
                       help='Number of workers for DataLoader')
    parser.add_argument('--post_process', default=False, type=lambda x: (str(x).lower() == 'true'),
                       help='Apply post-processing to predictions')
    parser.add_argument('--visualization', default=False, type=lambda x: (str(x).lower() == 'true'),
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Validate that the nnU-Net model file exists
    if not os.path.exists(args.nnunet_model_path):
        print(f"Error: nnU-Net model file not found: {args.nnunet_model_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("nnU-Net Integration with FSEFT Framework")
    print("=" * 60)
    print(f"Model path: {args.nnunet_model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Target organ: {args.organ}")
    print(f"Few-shot shots: {args.k}")
    print(f"Fine-tuning method: {args.method}")
    print(f"Decoder setting: {args.decoder}")
    print(f"Bottleneck setting: {args.bottleneck}")
    print("\nIMPORTANT NOTES:")
    print("- nnU-Net uses PlainConvUNet (6 stages) vs Framework UNet3D (4 stages)")
    print("- Only compatible layers will be transferred")
    print("- Patch size: 160x128x112 (nnU-Net) vs 96x96x96 (framework)")
    print("- Intensity range: 4.0-160.0 (nnU-Net) vs -175-250 (framework)")
    print("=" * 60)
    
    # Run the fine-tuning process
    try:
        process(args=args)
        print("\n" + "=" * 60)
        print("Fine-tuning completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError during fine-tuning: {e}")
        print("Please check your model path and configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
