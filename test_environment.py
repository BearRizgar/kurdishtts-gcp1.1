#!/usr/bin/env python3
"""
Test script to validate environment matches local Mac setup
"""
import sys
import platform

def test_environment():
    print("=== Environment Validation ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
        print(f"NumPy BLAS: {np.show_config()}")
    except ImportError:
        print("NumPy: NOT INSTALLED")
    
    try:
        import TTS
        print(f"TTS: {TTS.__version__}")
    except ImportError:
        print("TTS: NOT INSTALLED")
    
    try:
        import soundfile
        print(f"SoundFile: {soundfile.__version__}")
    except ImportError:
        print("SoundFile: NOT INSTALLED")
    
    try:
        import librosa
        print(f"Librosa: {librosa.__version__}")
    except ImportError:
        print("Librosa: NOT INSTALLED")
    
    print("=== Environment Check Complete ===")

if __name__ == "__main__":
    test_environment()