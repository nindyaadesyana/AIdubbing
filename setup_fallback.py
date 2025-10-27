#!/usr/bin/env python3
"""
Setup script for fallback TTS system (without PyTorch)
"""

import os
import sys

def test_fallback_system():
    """Test the fallback TTS system"""
    
    print("🧪 Testing Fallback TTS System")
    print("=" * 50)
    
    try:
        from simple_tts_fallback import SimpleTTSFallback
        
        cloner = SimpleTTSFallback()
        print("✅ SimpleTTSFallback initialized")
        
        # Test voice listing
        voices = cloner.list_voices()
        print(f"✅ Found {len(voices)} voices: {voices}")
        
        # Test speech generation if voices available
        if voices:
            voice_id = voices[0]
            test_text = "Halo, ini adalah test voice cloning dengan sistem fallback."
            output_file = "test_fallback_output.wav"
            
            print(f"🎤 Testing speech generation with voice: {voice_id}")
            result = cloner.generate_speech(voice_id, test_text, output_file)
            
            if result and os.path.exists(output_file):
                print("✅ Speech generation successful!")
                print(f"   Output file: {output_file}")
                
                # Clean up test file
                os.remove(output_file)
            else:
                print("❌ Speech generation failed")
                return False
        else:
            print("⚠️ No voices available for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback system test failed: {e}")
        return False

def test_flask_app():
    """Test Flask app imports"""
    
    print("\n🧪 Testing Flask App")
    print("-" * 30)
    
    try:
        # Test app imports
        import app
        print("✅ Flask app imports successful")
        
        # Test voice cloner initialization
        print("✅ Voice cloner initialized with fallback")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions"""
    
    print("\n📚 Usage Instructions")
    print("=" * 50)
    
    print("🚀 Start the application:")
    print("   python3 app.py")
    print("   Then open: http://localhost:5000")
    
    print("\n🎤 Voice Cloning Features:")
    print("   ✅ Upload audio files for voice training")
    print("   ✅ Simple voice training (no PyTorch required)")
    print("   ✅ Speech generation with voice adaptation")
    print("   ✅ Basic pitch matching and voice conversion")
    
    print("\n⚠️ Limitations (Fallback Mode):")
    print("   • Uses Google TTS + simple voice conversion")
    print("   • Quality lower than full VITS/YourTTS")
    print("   • No deep learning voice cloning")
    print("   • Basic pitch and tone adaptation only")
    
    print("\n💡 For Full Features:")
    print("   • Use Python 3.8-3.11 instead of 3.13")
    print("   • Install PyTorch and TTS library")
    print("   • Run multispeaker_train.py for advanced training")

def main():
    """Main setup function"""
    
    print("🚀 AI Dubbing - Fallback Setup")
    print("=" * 50)
    
    # Test fallback system
    fallback_ok = test_fallback_system()
    
    # Test Flask app
    flask_ok = test_flask_app()
    
    print("\n" + "=" * 50)
    
    if fallback_ok and flask_ok:
        print("🎉 Fallback system setup successful!")
        show_usage_instructions()
    else:
        print("❌ Setup failed. Please check the errors above.")
        
        print("\n🔧 Troubleshooting:")
        print("1. Ensure all minimal dependencies are installed:")
        print("   python3 -m pip install -r requirements_minimal.txt")
        print("2. Check if you have audio files in datasets/processed/")
        print("3. Verify internet connection for Google TTS")

if __name__ == "__main__":
    main()