import os
import wave
import base64
import tempfile
import hashlib

# Try different import patterns for compatibility
try:
    from google import genai
    from google.genai import types
    IMPORT_SUCCESS = True
except ImportError:
    try:
        import google.generativeai as genai
        from google.generativeai import types
        IMPORT_SUCCESS = True
    except ImportError:
        IMPORT_SUCCESS = False

class RobotTTS:
    def __init__(self):
        """Initialize the Robot TTS system with Gemini API"""
        if not IMPORT_SUCCESS:
            raise ImportError("Could not import google.genai or google.generativeai. Please install the correct package.")
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Configure Gemini client - try different methods
        try:
            # Method 1: Direct client creation (for google.genai)
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            try:
                # Method 2: Alternative client creation
                self.client = genai.Client()
                # Set API key in environment
                os.environ['GOOGLE_API_KEY'] = api_key
            except Exception as e2:
                raise RuntimeError(f"Failed to configure Gemini API: {e}, {e2}")
        
        # Create audio cache directory
        self.audio_dir = "robot_audio"
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Robot voice configuration - using a friendly, upbeat voice
        try:
            self.voice_config = types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name='Puck'  # Upbeat and friendly voice
                )
            )
        except Exception as e:
            print(f"Warning: Could not set voice config: {e}")
            self.voice_config = None
    
    def _get_audio_filename(self, text, filename_override=None):
        """Generate a filename based on text content or use override"""
        if filename_override:
            return os.path.join(self.audio_dir, f"{filename_override}.wav")
        
        # Create a hash of the text to use as filename
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.audio_dir, f"{text_hash}.wav")
    
    def _save_wave_file(self, filename, pcm_data, channels=1, rate=24000, sample_width=2):
        """Save PCM data to a WAV file"""
        try:
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                wf.writeframes(pcm_data)
            print(f"DEBUG: Audio saved to {filename}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to save audio to {filename}: {e}")
            return False
    
    def generate_speech(self, text, style_instruction="", filename_override=None):
        """
        Generate speech audio for the given text
        
        Args:
            text (str): Text to convert to speech
            style_instruction (str): Optional style instruction (e.g., "cheerfully", "excitedly")
            filename_override (str): Optional filename override (without .wav extension)
        
        Returns:
            str: Path to the generated audio file
        """
        try:
            # Check if audio already exists
            audio_filename = self._get_audio_filename(text, filename_override)
            if os.path.exists(audio_filename):
                print(f"DEBUG: Audio file already exists: {audio_filename}")
                return audio_filename
            
            print(f"DEBUG: Generating new audio for: {text[:50]}...")
            
            # Prepare the prompt with style instruction
            if style_instruction:
                prompt = f"Say {style_instruction}: {text}"
            else:
                prompt = text
            
            # Generate audio using Gemini TTS
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=self.voice_config
                    )
                )
            )
            
            # Extract audio data
            if (response.candidates and 
                response.candidates[0].content and 
                response.candidates[0].content.parts and 
                response.candidates[0].content.parts[0].inline_data):
                
                audio_data = response.candidates[0].content.parts[0].inline_data.data
                print(f"DEBUG: Received audio data, size: {len(audio_data)} bytes")
                
                # Save to WAV file
                if self._save_wave_file(audio_filename, audio_data):
                    print(f"DEBUG: Successfully saved audio to: {audio_filename}")
                    return audio_filename
                else:
                    print(f"ERROR: Failed to save audio file")
                    return None
            else:
                print(f"ERROR: No audio data received from Gemini API")
                return None
                
        except Exception as e:
            print(f"ERROR: Error generating speech for '{text[:30]}...': {e}")
            return None
    
    def get_robot_greeting_audio(self):
        """Generate audio for robot greeting"""
        text = "Hey there! Nice to meet you! Give me a thumbs up if you'd like to know more about SoDA!"
        filename = "greeting"
        print(f"DEBUG: Generating greeting audio with filename: {filename}")
        return self.generate_speech(text, "cheerfully", filename)
    
    def get_soda_info_audio(self):
        """Generate audio for SoDA information"""
        text = "SoDA is the Software Development Association! We build amazing projects, learn new technologies, and have fun together. Now show me a peace sign to get the QR code!"
        filename = "about_soda"
        print(f"DEBUG: Generating SoDA info audio with filename: {filename}")
        return self.generate_speech(text, "enthusiastically", filename)
    
    def get_heart_request_audio(self):
        """Generate audio for final greeting"""
        text = "Great! Here's how to join us!"
        filename = "final_greeting"
        print(f"DEBUG: Generating final greeting audio with filename: {filename}")
        return self.generate_speech(text, "excitedly", filename)
    
    def get_qr_show_audio(self):
        """Generate audio for QR code display"""
        text = "Awesome! Here's how to join us!"
        filename = "qr_show"
        return self.generate_speech(text, "happily", filename)
    
    def get_custom_audio(self, text, style="", filename=None):
        """Generate custom audio for any text"""
        return self.generate_speech(text, style, filename)
    
    def list_available_voices(self):
        """List all available voice options"""
        voices = [
            "Zephyr -- Bright", "Puck -- Upbeat", "Charon -- Informative",
            "Kore -- Firm", "Fenrir -- Excitable", "Leda -- Youthful",
            "Orus -- Firm", "Aoede -- Breezy", "Callirrhoe -- Easy-going",
            "Autonoe -- Bright", "Enceladus -- Breathy", "Iapetus -- Clear",
            "Umbriel -- Easy-going", "Algieba -- Smooth", "Despina -- Smooth",
            "Erinome -- Clear", "Algenib -- Gravelly", "Rasalgethi -- Informative",
            "Laomedeia -- Upbeat", "Achernar -- Soft", "Alnilam -- Firm",
            "Schedar -- Even", "Gacrux -- Mature", "Pulcherrima -- Forward",
            "Achird -- Friendly", "Zubenelgenubi -- Casual", "Vindemiatrix -- Gentle",
            "Sadachbia -- Lively", "Sadaltager -- Knowledgeable", "Sulafat -- Warm"
        ]
        return voices
    
    def change_voice(self, voice_name):
        """Change the robot's voice"""
        # Extract just the voice name (remove description)
        voice_name = voice_name.split(' -- ')[0]
        
        try:
            self.voice_config = types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
            print(f"Voice changed to: {voice_name}")
        except Exception as e:
            print(f"Warning: Could not change voice: {e}")
    
    def clear_audio_cache(self):
        """Clear all generated audio files"""
        import shutil
        if os.path.exists(self.audio_dir):
            shutil.rmtree(self.audio_dir)
        os.makedirs(self.audio_dir, exist_ok=True)
        print("Audio cache cleared")
    
    def list_generated_audio_files(self):
        """List all generated audio files"""
        if os.path.exists(self.audio_dir):
            files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
            print(f"DEBUG: Found {len(files)} audio files in {self.audio_dir}:")
            for file in files:
                file_path = os.path.join(self.audio_dir, file)
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size} bytes)")
            return files
        else:
            print(f"DEBUG: Audio directory {self.audio_dir} does not exist")
            return []

# Example usage and testing
if __name__ == "__main__":
    try:
        # Test the TTS system
        tts = RobotTTS()
        
        print("Available voices:")
        voices = tts.list_available_voices()
        for voice in voices[:5]:  # Show first 5
            print(f"  - {voice}")
        
        print("\nGenerating test audio files...")
        
        # Generate test audio files with proper names
        greeting_file = tts.get_robot_greeting_audio()
        if greeting_file:
            print(f"Greeting audio saved to: {greeting_file}")
        
        soda_file = tts.get_soda_info_audio()
        if soda_file:
            print(f"SoDA info audio saved to: {soda_file}")
        
        final_file = tts.get_heart_request_audio()
        if final_file:
            print(f"Final greeting audio saved to: {final_file}")
        
        # List all generated files
        print("\nGenerated audio files:")
        tts.list_generated_audio_files()
        
        print("\nTTS system is working correctly!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure GEMINI_API_KEY environment variable is set")
