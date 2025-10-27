#!/usr/bin/env python3
"""
Script to download and setup pretrained multi-speaker TTS models
"""

import os
import sys
from TTS.api import TTS

def download_pretrained_models():
    """Download pretrained multi-speaker models"""
    
    print("🔄 Setting up pretrained TTS models...")
    
    models_to_download = [
        {
            'name': 'YourTTS Multi-lingual',
            'model_name': 'tts_models/multilingual/multi-dataset/your_tts',
            'description': 'Multi-lingual multi-speaker model with voice cloning'
        },
        {
            'name': 'VITS Multi-speaker English',
            'model_name': 'tts_models/en/vctk/vits',
            'description': 'English multi-speaker VITS model'
        }
    ]
    
    successful_downloads = []
    
    for model_info in models_to_download:
        try:
            print(f"\n📥 Downloading {model_info['name']}...")
            print(f"   Description: {model_info['description']}")
            
            # Initialize TTS with the model (this will download it)
            tts = TTS(model_name=model_info['model_name'])
            
            print(f"✅ Successfully downloaded: {model_info['name']}")
            successful_downloads.append(model_info)
            
            # Test the model
            print("🧪 Testing model...")
            test_output = "test_output.wav"
            tts.tts_to_file(text="Hello, this is a test.", file_path=test_output)
            
            if os.path.exists(test_output):
                os.remove(test_output)
                print("✅ Model test successful")
            
        except Exception as e:
            print(f"❌ Failed to download {model_info['name']}: {e}")
            continue
    
    print(f"\n🎉 Setup complete! Downloaded {len(successful_downloads)} models:")
    for model in successful_downloads:
        print(f"   ✅ {model['name']}")
    
    if not successful_downloads:
        print("⚠️ No models were downloaded successfully.")
        print("Please check your internet connection and try again.")
        return False
    
    return True

def verify_dependencies():
    """Verify that all required dependencies are installed"""
    
    print("🔍 Verifying dependencies...")
    
    required_packages = [
        'torch',
        'torchaudio', 
        'TTS',
        'librosa',
        'soundfile',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies verified!")
    return True

def main():
    """Main setup function"""
    
    print("🚀 AI Dubbing - Pretrained Models Setup")
    print("=" * 50)
    
    # Verify dependencies first
    if not verify_dependencies():
        print("\n❌ Please install missing dependencies before continuing.")
        sys.exit(1)
    
    # Download pretrained models
    if download_pretrained_models():
        print("\n🎉 Setup completed successfully!")
        print("\nYou can now use the multi-speaker voice cloning features:")
        print("1. Upload audio files for voice cloning")
        print("2. Train custom models with speaker embedding")
        print("3. Generate speech with voice cloning")
        
        print("\n📚 Available models:")
        print("- YourTTS: Multi-lingual voice cloning")
        print("- VITS: High-quality multi-speaker synthesis")
        
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()