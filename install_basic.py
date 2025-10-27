#!/usr/bin/env python3
"""
Basic installation without PyTorch for Python 3.13
"""

import subprocess
import sys

def install_basic():
    print("🚀 Installing basic dependencies (without PyTorch)...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_basic.txt"], check=True)
        print("✅ Basic dependencies installed")
        
        print("\n📝 Manual PyTorch installation needed:")
        print("For Python 3.13, install PyTorch manually:")
        print("pip install torch torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cpu")
        print("\nOr use conda:")
        print("conda install pytorch torchaudio cpuonly -c pytorch-nightly")
        
        return True
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

if __name__ == "__main__":
    install_basic()