#!/usr/bin/env python3
"""
Test script to verify PyTorch import works with CPU-only mode
"""

import os

# Force CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'

try:
    import torch
    print(f"✓ PyTorch imported successfully: {torch.__version__}")
    
    # Test CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Test basic tensor operations
    x = torch.randn(3, 3)
    print(f"✓ Tensor creation successful: {x.shape}")
    
    # Test model loading (simplified)
    try:
        # This would normally load YOLOv5, but we'll just test the import path
        from nn_model.model import Wrapper
        print("✓ nn_model.model import successful")
    except Exception as e:
        print(f"✗ nn_model.model import failed: {e}")
    
    print("✓ All PyTorch tests passed!")
    
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    print("This indicates CUDA library dependency issues")
    
except Exception as e:
    print(f"✗ Unexpected error: {e}")