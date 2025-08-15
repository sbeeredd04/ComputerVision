#!/usr/bin/env python3
"""
Demo script for Robot TTS functionality
This script demonstrates the text-to-speech capabilities of the SoDA robot.
"""

import os
import time
from tts_utils import RobotTTS

def main():
    print("🤖 SoDA Robot TTS Demo")
    print("=" * 50)
    
    # Set API key
    api_key = "AIzaSyD4nmZwglvux1j_PTeAmAYhhlxHgfpTTmU"
    os.environ['GEMINI_API_KEY'] = api_key
    
    try:
        # Initialize TTS system
        print("🎤 Initializing TTS system...")
        tts = RobotTTS()
        print("✅ TTS system ready!")
        
        # Show available voices
        print(f"\n🎭 Available voices: {len(tts.list_available_voices())}")
        print("   Sample voices:")
        voices = tts.list_available_voices()
        for voice in voices[:5]:
            print(f"   - {voice}")
        
        # Demo 1: Robot Greeting
        print("\n🎵 Demo 1: Robot Greeting")
        print("   Text: 'Hey there! Nice to meet you! Give me a thumbs up if you'd like to know more about SoDA!'")
        print("   Style: cheerfully")
        
        greeting_file = tts.get_robot_greeting_audio()
        if greeting_file:
            print(f"   ✅ Audio generated: {os.path.basename(greeting_file)}")
            print(f"   📁 File size: {os.path.getsize(greeting_file)} bytes")
        else:
            print("   ❌ Failed to generate audio")
        
        # Demo 2: SoDA Information
        print("\n🎵 Demo 2: SoDA Information")
        print("   Text: 'SoDA is the Software Development Association! We build amazing projects, learn new technologies, and have fun together. Want to join us?'")
        print("   Style: enthusiastically")
        
        soda_file = tts.get_soda_info_audio()
        if soda_file:
            print(f"   ✅ Audio generated: {os.path.basename(soda_file)}")
            print(f"   📁 File size: {os.path.getsize(soda_file)} bytes")
        else:
            print("   ❌ Failed to generate audio")
        
        # Demo 3: Heart Gesture Request
        print("\n🎵 Demo 3: Heart Gesture Request")
        print("   Text: 'If you like what you see, give me a heart gesture!'")
        print("   Style: excitedly")
        
        heart_file = tts.get_heart_request_audio()
        if heart_file:
            print(f"   ✅ Audio generated: {os.path.basename(heart_file)}")
            print(f"   📁 File size: {os.path.getsize(heart_file)} bytes")
        else:
            print("   ❌ Failed to generate audio")
        
        # Demo 4: Custom TTS
        print("\n🎵 Demo 4: Custom TTS")
        custom_text = "Welcome to the future of interactive robotics!"
        custom_style = "futuristically"
        print(f"   Text: '{custom_text}'")
        print(f"   Style: {custom_style}")
        
        custom_file = tts.get_custom_audio(custom_text, custom_style)
        if custom_file:
            print(f"   ✅ Audio generated: {os.path.basename(custom_file)}")
            print(f"   📁 File size: {os.path.getsize(custom_file)} bytes")
        else:
            print("   ❌ Failed to generate audio")
        
        # Demo 5: Voice Change
        print("\n🎭 Demo 5: Voice Change")
        print("   Changing from Puck (Upbeat) to Kore (Firm)...")
        tts.change_voice("Kore -- Firm")
        
        # Generate a test with new voice
        test_file = tts.get_custom_audio("This is a test with a different voice!", "firmly")
        if test_file:
            print(f"   ✅ New voice audio: {os.path.basename(test_file)}")
        
        # Change back to Puck
        tts.change_voice("Puck -- Upbeat")
        print("   ✅ Voice changed back to Puck")
        
        # Summary
        print("\n📊 Summary")
        print("=" * 50)
        audio_dir = tts.audio_dir
        if os.path.exists(audio_dir):
            files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
            total_size = sum(os.path.getsize(os.path.join(audio_dir, f)) for f in files)
            print(f"   📁 Total audio files: {len(files)}")
            print(f"   💾 Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
            print(f"   🎵 Files:")
            for file in files:
                file_path = os.path.join(audio_dir, file)
                size = os.path.getsize(file_path)
                print(f"      - {file} ({size:,} bytes)")
        
        print("\n🎉 Demo completed successfully!")
        print("\n🚀 To run the full Streamlit app:")
        print("   export GEMINI_API_KEY='your_api_key'")
        print("   streamlit run streamlit_soda_advanced.py")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
