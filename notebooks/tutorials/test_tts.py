#!/usr/bin/env python3
"""
Test script for Robot TTS functionality
Run this to verify that the Gemini TTS integration is working correctly.
"""

import os
import sys

def test_tts():
    """Test the TTS functionality"""
    print("🤖 Testing Robot TTS System...")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY='your_key_here'")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    try:
        # Import TTS module
        from tts_utils import RobotTTS
        print("✅ TTS module imported successfully")
        
        # Initialize TTS system
        tts = RobotTTS()
        print("✅ TTS system initialized")
        
        # List available voices
        voices = tts.list_available_voices()
        print(f"✅ Found {len(voices)} available voices")
        print("   Sample voices:")
        for voice in voices[:5]:
            print(f"   - {voice}")
        
        # Test audio generation
        print("\n🎵 Testing audio generation...")
        
        # Test greeting audio
        print("   Generating greeting audio...")
        greeting_file = tts.get_robot_greeting_audio()
        if greeting_file and os.path.exists(greeting_file):
            print(f"   ✅ Greeting audio: {os.path.basename(greeting_file)}")
        else:
            print("   ❌ Failed to generate greeting audio")
            return False
        
        # Test SoDA info audio
        print("   Generating SoDA info audio...")
        soda_file = tts.get_soda_info_audio()
        if soda_file and os.path.exists(soda_file):
            print(f"   ✅ SoDA info audio: {os.path.basename(soda_file)}")
        else:
            print("   ❌ Failed to generate SoDA info audio")
            return False
        
        # Test custom audio
        print("   Generating custom audio...")
        custom_file = tts.get_custom_audio("Hello, this is a test!", "cheerfully")
        if custom_file and os.path.exists(custom_file):
            print(f"   ✅ Custom audio: {os.path.basename(custom_file)}")
        else:
            print("   ❌ Failed to generate custom audio")
            return False
        
        # Test voice change
        print("\n🎭 Testing voice change...")
        original_voice = tts.voice_config.prebuilt_voice_config.voice_name
        tts.change_voice("Kore -- Firm")
        new_voice = tts.voice_config.prebuilt_voice_config.voice_name
        if new_voice == "Kore":
            print("   ✅ Voice changed successfully")
        else:
            print("   ❌ Voice change failed")
            return False
        
        # Restore original voice
        tts.change_voice(f"{original_voice} -- Upbeat")
        
        # List generated files
        print("\n📁 Generated audio files:")
        audio_dir = tts.audio_dir
        if os.path.exists(audio_dir):
            files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
            for file in files:
                file_path = os.path.join(audio_dir, file)
                size = os.path.getsize(file_path)
                print(f"   - {file} ({size} bytes)")
        
        print("\n🎉 All tests passed! TTS system is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed the requirements: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_tts()
    sys.exit(0 if success else 1)
