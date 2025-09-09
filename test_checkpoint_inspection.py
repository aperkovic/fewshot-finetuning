#!/usr/bin/env python3
"""
Test script for checkpoint inspection workflow.

This script demonstrates how to use the checkpoint inspection tools
to analyze a model and generate a class template.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False


def main():
    print("Checkpoint Inspection Workflow Test")
    print("=" * 50)
    
    # Check if we have any checkpoint files to test with
    checkpoint_dir = Path("./models/pretrained_weights")
    if not checkpoint_dir.exists():
        print(f"Warning: {checkpoint_dir} does not exist. Creating it...")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for existing checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoint_files:
        print("No checkpoint files found in ./models/pretrained_weights/")
        print("Please add a checkpoint file to test the inspection workflow.")
        print("\nExample usage:")
        print("  python inspect_checkpoint.py --checkpoint_path /path/to/your/model.pth")
        print("  python generate_model_class.py --analysis_file checkpoint_analysis.json --output_file MyModel.py --class_name MyModel")
        return
    
    # Test with the first available checkpoint
    checkpoint_path = checkpoint_files[0]
    print(f"Testing with checkpoint: {checkpoint_path}")
    
    # Step 1: Inspect the checkpoint
    inspect_cmd = [
        sys.executable, "inspect_checkpoint.py",
        "--checkpoint_path", str(checkpoint_path),
        "--output_log", "test_analysis.log",
        "--output_json", "test_analysis.json"
    ]
    
    if not run_command(inspect_cmd, "Step 1: Inspecting checkpoint"):
        print("Checkpoint inspection failed. Please check the error messages above.")
        return
    
    # Step 2: Generate a model class
    if os.path.exists("test_analysis.json"):
        generate_cmd = [
            sys.executable, "generate_model_class.py",
            "--analysis_file", "test_analysis.json",
            "--output_file", "GeneratedModel.py",
            "--class_name", "GeneratedModel",
            "--architecture_type", "generic"
        ]
        
        if not run_command(generate_cmd, "Step 2: Generating model class"):
            print("Model class generation failed. Please check the error messages above.")
            return
        
        print("\n" + "="*60)
        print("SUCCESS! Workflow completed successfully.")
        print("="*60)
        print("Generated files:")
        print("  - test_analysis.log: Detailed analysis log")
        print("  - test_analysis.json: Structured analysis data")
        print("  - GeneratedModel.py: Template model class")
        print("\nNext steps:")
        print("  1. Review the analysis log to understand the model structure")
        print("  2. Implement the actual architecture in GeneratedModel.py")
        print("  3. Test the model with your data")
    else:
        print("Analysis JSON file not found. Check the inspection step for errors.")


if __name__ == "__main__":
    main()
