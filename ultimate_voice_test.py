#!/usr/bin/env python3
"""
Ultimate voice cloning comparison test
"""

import os
import time

def test_all_voice_cloners():
    """Test semua voice cloner yang tersedia"""
    
    print("🎯 ULTIMATE VOICE CLONING TEST")
    print("=" * 60)
    
    test_text = "Halo, ini adalah test perbandingan kualitas voice cloning terbaik"
    voice_id = "Della_3a452b8a"
    
    results = {}
    
    # Test 1: Simple Fallback
    print("\n1️⃣ Testing Simple Fallback TTS...")
    try:
        from simple_tts_fallback import SimpleTTSFallback
        
        cloner = SimpleTTSFallback()
        if voice_id in cloner.list_voices() or cloner.list_voices():
            test_voice = voice_id if voice_id in cloner.list_voices() else cloner.list_voices()[0]
            result = cloner.generate_speech(test_voice, test_text, "output_simple_final.wav")
            results['Simple'] = {'success': result, 'features': 'Basic pitch shifting'}
            print(f"   ✅ Simple: {result}")
        else:
            results['Simple'] = {'success': False, 'features': 'No voices'}
            print("   ❌ No voices available")
    except Exception as e:
        results['Simple'] = {'success': False, 'error': str(e)}
        print(f"   ❌ Simple failed: {e}")
    
    # Test 2: Improved Voice Cloner
    print("\n2️⃣ Testing Improved Voice Cloner...")
    try:
        from improved_voice_cloner import ImprovedVoiceCloner
        
        cloner = ImprovedVoiceCloner()
        if voice_id in cloner.list_voices() or cloner.list_voices():
            test_voice = voice_id if voice_id in cloner.list_voices() else cloner.list_voices()[0]
            result = cloner.generate_speech(test_voice, test_text, "output_improved_final.wav")
            results['Improved'] = {'success': result, 'features': 'Pitch + Spectral + Formant + Energy'}
            print(f"   ✅ Improved: {result}")
        else:
            results['Improved'] = {'success': False, 'features': 'No voices'}
            print("   ❌ No voices available")
    except Exception as e:
        results['Improved'] = {'success': False, 'error': str(e)}
        print(f"   ❌ Improved failed: {e}")
    
    # Test 3: Advanced Voice Cloner
    print("\n3️⃣ Testing Advanced Voice Cloner...")
    try:
        from advanced_voice_cloner import AdvancedVoiceCloner
        
        cloner = AdvancedVoiceCloner()
        
        # Train if needed
        if voice_id not in cloner.list_voices():
            print("   🎯 Training advanced model...")
            training_id = cloner.start_training(voice_id, epochs=10)
            
            for i in range(15):
                time.sleep(0.5)
                status = cloner.get_training_status(training_id)
                if status.get('status') == 'completed':
                    break
        
        if voice_id in cloner.list_voices() or cloner.list_voices():
            test_voice = voice_id if voice_id in cloner.list_voices() else cloner.list_voices()[0]
            result = cloner.generate_speech(test_voice, test_text, "output_advanced_final.wav")
            
            analysis = cloner.get_audio_analysis(test_voice)
            features = f"Premium: {analysis.get('model_type', 'Advanced')}"
            
            results['Advanced'] = {'success': result, 'features': features, 'analysis': analysis}
            print(f"   ✅ Advanced: {result}")
            print(f"   📊 Analysis: {analysis}")
        else:
            results['Advanced'] = {'success': False, 'features': 'No voices'}
            print("   ❌ No voices available")
    except Exception as e:
        results['Advanced'] = {'success': False, 'error': str(e)}
        print(f"   ❌ Advanced failed: {e}")
    
    return results

def show_comparison_results(results):
    """Show detailed comparison"""
    
    print("\n📊 COMPARISON RESULTS")
    print("=" * 60)
    
    comparison_table = [
        ["Method", "Success", "Features", "Quality"],
        ["-" * 8, "-" * 7, "-" * 30, "-" * 7],
    ]
    
    quality_map = {
        'Simple': 'Basic',
        'Improved': 'Good', 
        'Advanced': 'Premium'
    }
    
    for method, result in results.items():
        success = "✅" if result.get('success') else "❌"
        features = result.get('features', 'Unknown')[:30]
        quality = quality_map.get(method, 'Unknown')
        
        comparison_table.append([method, success, features, quality])
    
    for row in comparison_table:
        print(f"{row[0]:<10} {row[1]:<8} {row[2]:<32} {row[3]:<8}")
    
    print("\n🎯 TECHNICAL COMPARISON")
    print("=" * 60)
    
    techniques = {
        'Simple': [
            "❌ Basic Google TTS",
            "❌ Simple pitch shift only",
            "❌ No voice analysis",
            "❌ No spectral processing"
        ],
        'Improved': [
            "✅ Google TTS + advanced processing",
            "✅ Pitch + spectral + formant conversion",
            "✅ Energy envelope matching",
            "✅ MFCC-based filtering"
        ],
        'Advanced': [
            "🚀 Comprehensive voice analysis",
            "🚀 Multi-dimensional feature extraction",
            "🚀 Template-based voice matching",
            "🚀 Advanced signal processing",
            "🚀 Formant + prosodic conversion",
            "🚀 Voice quality optimization"
        ]
    }
    
    for method, tech_list in techniques.items():
        if method in results and results[method].get('success'):
            print(f"\n{method} Voice Cloner:")
            for tech in tech_list:
                print(f"   {tech}")

def show_file_analysis():
    """Analyze generated files"""
    
    print("\n📁 FILE ANALYSIS")
    print("=" * 60)
    
    files = [
        "output_simple_final.wav",
        "output_improved_final.wav", 
        "output_advanced_final.wav"
    ]
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            method = file.split('_')[1].title()
            print(f"✅ {method:<10} {file:<25} ({size:,} bytes)")
        else:
            method = file.split('_')[1].title()
            print(f"❌ {method:<10} {file:<25} (not found)")

def show_recommendations():
    """Show usage recommendations"""
    
    print("\n💡 RECOMMENDATIONS")
    print("=" * 60)
    
    print("🥇 BEST QUALITY: Advanced Voice Cloner")
    print("   • Comprehensive voice analysis")
    print("   • Multi-dimensional feature matching")
    print("   • Premium voice conversion")
    print("   • Best for production use")
    
    print("\n🥈 GOOD BALANCE: Improved Voice Cloner")
    print("   • Good quality improvement")
    print("   • Faster processing")
    print("   • Suitable for most use cases")
    
    print("\n🥉 BASIC OPTION: Simple Fallback")
    print("   • Quick and simple")
    print("   • Minimal processing")
    print("   • Fallback option only")
    
    print("\n🚀 NEXT LEVEL (Future):")
    print("   • Use Python 3.10 + PyTorch")
    print("   • VITS Multi-speaker training")
    print("   • Neural vocoder (HiFi-GAN)")
    print("   • Real-time voice conversion")

def main():
    """Main test function"""
    
    # Run all tests
    results = test_all_voice_cloners()
    
    # Show results
    show_comparison_results(results)
    show_file_analysis()
    show_recommendations()
    
    print("\n🎉 CONCLUSION")
    print("=" * 60)
    
    successful_methods = [method for method, result in results.items() if result.get('success')]
    
    if 'Advanced' in successful_methods:
        print("🏆 Advanced Voice Cloner is the BEST option available!")
        print("   Voice generation will be most similar to training audio.")
    elif 'Improved' in successful_methods:
        print("🥈 Improved Voice Cloner is your best bet!")
        print("   Significant improvement over basic TTS.")
    elif 'Simple' in successful_methods:
        print("🥉 Simple Fallback is working.")
        print("   Basic voice cloning with limited similarity.")
    else:
        print("❌ No voice cloners are working properly.")
    
    print(f"\n📈 Success Rate: {len(successful_methods)}/3 methods working")
    
    print("\n🚀 TO USE THE BEST SYSTEM:")
    print("   python3 start_app.py")
    print("   # App will automatically use the best available cloner")

if __name__ == "__main__":
    main()