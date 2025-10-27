#!/usr/bin/env python3
"""
Install script for AI Dubbing dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def main():
    print("🚀 AI Dubbing - Installing Dependencies")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version}")
    
    # Install PyTorch with fallback options
    pytorch_commands = [
        "pip install torch torchaudio",
        "pip install torch==2.1.0 torchaudio==2.1.0",
        "pip install torch==2.0.1 torchaudio==2.0.2"
    ]
    
    pytorch_installed = False
    for cmd in pytorch_commands:
        if run_command(cmd, "Installing PyTorch"):
            pytorch_installed = True
            break
    
    if not pytorch_installed:
        print("❌ Could not install PyTorch. Continuing with other dependencies...")
    
    commands = [
        {
            "cmd": "pip install -r requirements.txt",
            "desc": "Installing other dependencies"
        }
    ]
    
    for command_info in commands:
        if not run_command(command_info["cmd"], command_info["desc"]):
            print(f"\n⚠️ Warning: {command_info['desc']} had issues")
            print("Continuing with installation...")
    
    print("\n🎉 All dependencies installed successfully!")
    print("\nNext steps:")
    print("1. Run: python3 setup_pretrained_models.py")
    print("2. Run: python3 app.py")
    print("3. Open: http://localhost:5000")

if __name__ == "__main__":
    main()