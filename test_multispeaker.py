#!/usr/bin/env python3
"""
Test script for multi-speaker voice cloning
"""

import os
import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('torchaudio', 'TorchAudio'),
        ('TTS', 'Coqui TTS'),
        ('librosa', 'Librosa'),
        ('soundfile', 'SoundFile'),
        ('numpy', 'NumPy'),
        ('flask', 'Flask')
    ]
    
    failed_imports = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0

def test_multispeaker_cloner():
    """Test the multi-speaker voice cloner"""
    print("\n🧪 Testing MultiSpeakerVoiceCloner...")
    
    try:
        from multispeaker_voice_cloner import MultiSpeakerVoiceCloner
        
        cloner = MultiSpeakerVoiceCloner()
        print("✅ MultiSpeakerVoiceCloner initialized")
        
        # Test voice listing
        voices = cloner.list_voices()
        print(f"✅ Found {len(voices)} voices: {voices}")
        
        # Test voice info
        for voice_id in voices:
            info = cloner.get_voice_info(voice_id)
            if info:
                print(f"✅ Voice info for {voice_id}: {info.get('model_type', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ MultiSpeakerVoiceCloner test failed: {e}")
        return False

def test_dataset():
    """Test if dataset is available"""
    print("\n🧪 Testing dataset...")
    
    dataset_dirs = [
        "datasets/processed/Della_3a452b8a",
        "datasets/processed/indira_222056cf"
    ]
    
    found_datasets = []
    
    for dataset_dir in dataset_dirs:
        if os.path.exists(dataset_dir):
            metadata_file = os.path.join(dataset_dir, "metadata.csv")
            clips_dir = os.path.join(dataset_dir, "clips")
            
            if os.path.exists(metadata_file) and os.path.exists(clips_dir):
                clip_count = len([f for f in os.listdir(clips_dir) if f.endswith('.wav')])
                print(f"✅ Dataset {os.path.basename(dataset_dir)}: {clip_count} clips")
                found_datasets.append(dataset_dir)
            else:
                print(f"⚠️ Dataset {os.path.basename(dataset_dir)}: incomplete")
        else:
            print(f"❌ Dataset {os.path.basename(dataset_dir)}: not found")
    
    return len(found_datasets) > 0

def test_pretrained_models():
    """Test pretrained model availability"""
    print("\n🧪 Testing pretrained models...")
    
    try:
        from TTS.api import TTS
        
        models_to_test = [
            "tts_models/multilingual/multi-dataset/your_tts",
            "tts_models/en/vctk/vits"
        ]
        
        available_models = []
        
        for model_name in models_to_test:
            try:
                tts = TTS(model_name=model_name)
                print(f"✅ {model_name}")
                available_models.append(model_name)
            except Exception as e:
                print(f"❌ {model_name}: {e}")
        
        return len(available_models) > 0
        
    except Exception as e:
        print(f"❌ Pretrained model test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 AI Dubbing - Multi-Speaker Test Suite")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Dataset Availability", test_dataset),
        ("MultiSpeaker Cloner", test_multispeaker_cloner),
        ("Pretrained Models", test_pretrained_models)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed_tests += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Multi-speaker voice cloning is ready.")
        print("\nYou can now:")
        print("1. Start the app: python3 app.py")
        print("2. Train models with speaker embedding")
        print("3. Generate speech with voice cloning")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        
        if passed_tests == 0:
            print("\n💡 Quick fixes:")
            print("1. Install dependencies: python3 install_dependencies.py")
            print("2. Setup models: python3 setup_pretrained_models.py")

if __name__ == "__main__":
    main()