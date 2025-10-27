#!/usr/bin/env python3
"""
AI Voice Cloning - Dubbing App
Aplikasi web untuk voice cloning menggunakan TTS library
"""

import os
import sys

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import flask
        import librosa
        import pydub
        print("✅ All dependencies are installed")
        print("🎤 Using simple voice cloning implementation")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        'uploads',
        'models', 
        'outputs',
        'datasets/processed',
        'datasets/raw'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory created: {directory}")

def main():
    print("🎤 AI Voice Cloning - Dubbing App")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Import and run app
    try:
        from app import app
        print("\n🚀 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5001")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 40)
        
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()