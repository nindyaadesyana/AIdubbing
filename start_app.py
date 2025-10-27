#!/usr/bin/env python3
"""
Startup script for AI Dubbing app
"""

import os
import sys
import webbrowser
import time
from threading import Timer

def open_browser():
    """Open browser after a delay"""
    time.sleep(2)  # Wait for Flask to start
    webbrowser.open('http://localhost:5000')

def main():
    print("🚀 Starting AI Dubbing Application")
    print("=" * 50)
    
    # Check if we're using fallback
    try:
        import torch
        print("✅ PyTorch available - Full features enabled")
    except ImportError:
        print("⚠️ Using fallback TTS (PyTorch not available)")
        print("   Features: Basic voice cloning with pitch adaptation")
    
    print("\n🌐 Starting web server...")
    print("   URL: http://localhost:5000")
    print("   Press Ctrl+C to stop")
    
    # Open browser automatically
    Timer(2.0, open_browser).start()
    
    # Start Flask app
    try:
        import app
        app.app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()