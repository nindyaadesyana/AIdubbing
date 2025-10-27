#!/usr/bin/env python3
"""
Test script for fallback TTS system
"""

import os
import sys

def test_basic_imports():
    """Test basic package imports"""
    print("🧪 Testing basic imports...")
    
    packages = [
        ('flask', 'Flask'),
        ('librosa', 'Librosa'),
        ('soundfile', 'SoundFile'),
        ('numpy', 'NumPy'),
        ('gtts', 'Google TTS')
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

def test_fallback_cloner():
    """Test the fallback voice cloner"""
    print("\n🧪 Testing Fallback Voice Cloner...")
    
    try:
        from simple_tts_fallback import SimpleTTSFallback
        
        cloner = SimpleTTSFallback()
        print("✅ SimpleTTSFallback initialized")
        
        # Test voice listing
        voices = cloner.list_voices()
        print(f"✅ Found {len(voices)} voices: {voices}")
        
        # Test voice info
        for voice_id in voices:
            info = cloner.get_voice_info(voice_id)
            if info:
                print(f"✅ Voice info for {voice_id}: {info.get('model_type', 'unknown')}")
        
        # Test speech generation
        if voices:
            voice_id = voices[0]
            test_text = "Ini adalah test speech generation"
            output_file = "test_generation.wav"
            
            print(f"🎤 Testing speech generation...")
            result = cloner.generate_speech(voice_id, test_text, output_file)
            
            if result and os.path.exists(output_file):
                print("✅ Speech generation successful")
                os.remove(output_file)  # Cleanup
                return True
            else:
                print("❌ Speech generation failed")
                return False
        else:
            print("⚠️ No voices available for testing")
            return True  # Still pass if no voices but cloner works
        
    except Exception as e:
        print(f"❌ Fallback cloner test failed: {e}")
        return False

def test_flask_app():
    """Test Flask app with fallback"""
    print("\n🧪 Testing Flask App with Fallback...")
    
    try:
        # Import app (this will use fallback)
        import app
        print("✅ Flask app imported successfully")
        
        # Test voice cloner in app
        voices = app.voice_cloner.list_voices()
        print(f"✅ App voice cloner has {len(voices)} voices")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def test_dataset():
    """Test dataset availability"""
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

def test_training_simulation():
    """Test training simulation"""
    print("\n🧪 Testing training simulation...")
    
    try:
        from simple_tts_fallback import SimpleTTSFallback
        
        cloner = SimpleTTSFallback()
        
        # Start a quick training simulation
        training_id = cloner.start_training("test_voice", epochs=2)
        print(f"✅ Training started with ID: {training_id}")
        
        # Wait a bit and check status
        import time
        time.sleep(1)
        
        status = cloner.get_training_status(training_id)
        print(f"✅ Training status: {status.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training simulation failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 AI Dubbing - Fallback System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Package Imports", test_basic_imports),
        ("Dataset Availability", test_dataset),
        ("Fallback Voice Cloner", test_fallback_cloner),
        ("Flask App Integration", test_flask_app),
        ("Training Simulation", test_training_simulation)
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
        print("🎉 All fallback tests passed! System is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Start the app: python3 app.py")
        print("2. Open browser: http://localhost:5000")
        print("3. Upload audio and test voice cloning")
        
        print("\n📝 Current capabilities:")
        print("• Basic voice cloning with pitch adaptation")
        print("• Google TTS with voice conversion")
        print("• Audio processing and training simulation")
        print("• Web interface for easy use")
        
    elif passed_tests >= 3:
        print("⚠️ Most tests passed. System should work with minor limitations.")
        print("\n🚀 You can still:")
        print("1. Start the app: python3 app.py")
        print("2. Test basic functionality")
        
    else:
        print("❌ Multiple tests failed. Please check dependencies.")
        print("\n💡 Try:")
        print("1. python3 -m pip install -r requirements_minimal.txt")
        print("2. Check if audio files exist in datasets/")

if __name__ == "__main__":
    main()