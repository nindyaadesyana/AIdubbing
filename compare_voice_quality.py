#!/usr/bin/env python3
"""
Script untuk membandingkan kualitas voice generation
"""

import os
import time

def test_voice_cloners():
    """Test dan bandingkan voice cloners"""
    
    print("🔬 Perbandingan Kualitas Voice Generation")
    print("=" * 60)
    
    test_text = "Halo, ini adalah test perbandingan kualitas voice cloning"
    
    # Test 1: Simple Fallback
    print("\n1️⃣ Testing Simple Fallback TTS...")
    try:
        from simple_tts_fallback import SimpleTTSFallback
        
        simple_cloner = SimpleTTSFallback()
        voices = simple_cloner.list_voices()
        
        if voices:
            voice_id = voices[0]
            result = simple_cloner.generate_speech(voice_id, test_text, "output_simple.wav")
            print(f"   ✅ Simple TTS: {result}")
            print(f"   📊 Features: Basic pitch shifting only")
        else:
            print("   ❌ No voices available")
            
    except Exception as e:
        print(f"   ❌ Simple TTS failed: {e}")
    
    # Test 2: Improved Voice Cloner
    print("\n2️⃣ Testing Improved Voice Cloner...")
    try:
        from improved_voice_cloner import ImprovedVoiceCloner
        
        improved_cloner = ImprovedVoiceCloner()
        voices = improved_cloner.list_voices()
        
        if voices:
            voice_id = voices[0]
            
            # Get voice analysis
            analysis = improved_cloner.get_audio_analysis(voice_id)
            print(f"   📊 Voice Analysis: {analysis}")
            
            result = improved_cloner.generate_speech(voice_id, test_text, "output_improved.wav")
            print(f"   ✅ Improved TTS: {result}")
            print(f"   📊 Features: Pitch + Spectral + Formant + Energy matching")
        else:
            print("   ❌ No voices available")
            
    except Exception as e:
        print(f"   ❌ Improved TTS failed: {e}")
    
    # Show improvements
    print("\n🎯 Perbaikan yang Diterapkan:")
    print("-" * 40)
    
    improvements = [
        "✅ Ekstraksi fitur suara komprehensif (pitch, formant, spektral)",
        "✅ Konversi pitch yang lebih akurat",
        "✅ Matching spectral envelope",
        "✅ Konversi formant dasar",
        "✅ Matching energy envelope dari reference audio",
        "✅ Filtering adaptif berdasarkan karakteristik suara",
        "✅ Menggunakan multiple reference clips",
        "✅ Analisis MFCC untuk karakteristik spektral"
    ]
    
    for improvement in improvements:
        print(f"   {improvement}")
    
    print("\n📈 Hasil yang Diharapkan:")
    print("-" * 40)
    print("   🎵 Pitch lebih sesuai dengan suara target")
    print("   🔊 Timbre dan resonansi lebih mirip")
    print("   📊 Karakteristik spektral lebih cocok")
    print("   ⚡ Energy dan dinamika lebih natural")
    
    # Check output files
    print("\n📁 File Output:")
    print("-" * 40)
    
    files = ["output_simple.wav", "output_improved.wav"]
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file} ({size} bytes)")
        else:
            print(f"   ❌ {file} not found")

def show_technical_details():
    """Show technical details of improvements"""
    
    print("\n🔧 Detail Teknis Perbaikan:")
    print("=" * 60)
    
    techniques = [
        {
            "name": "Pitch Conversion",
            "old": "Simple pitch shift (-4 to +4 semitones)",
            "new": "Adaptive pitch matching based on voice statistics"
        },
        {
            "name": "Spectral Processing", 
            "old": "No spectral modification",
            "new": "Spectral centroid matching + adaptive filtering"
        },
        {
            "name": "Formant Processing",
            "old": "No formant processing",
            "new": "Basic formant shifting using multi-rate pitch shift"
        },
        {
            "name": "Energy Matching",
            "old": "Simple normalization",
            "new": "Energy envelope matching from reference audio"
        },
        {
            "name": "Voice Analysis",
            "old": "No voice analysis",
            "new": "Comprehensive feature extraction (MFCC, pitch stats, etc)"
        }
    ]
    
    for i, tech in enumerate(techniques, 1):
        print(f"\n{i}. {tech['name']}:")
        print(f"   ❌ Sebelum: {tech['old']}")
        print(f"   ✅ Sesudah: {tech['new']}")

if __name__ == "__main__":
    test_voice_cloners()
    show_technical_details()
    
    print("\n🚀 Cara Menggunakan:")
    print("=" * 60)
    print("1. python3 start_app.py")
    print("2. Buka http://localhost:5000")
    print("3. Upload audio dan test voice cloning")
    print("4. Bandingkan kualitas dengan sebelumnya")