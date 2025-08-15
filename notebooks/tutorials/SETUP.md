# 🚀 Quick Setup Guide for SoDA Robot TTS

## ⚡ **5-Minute Setup**

### 1. **Get Your Gemini API Key**
- Go to [Google AI Studio](https://aistudio.google.com/)
- Sign in with your Google account
- Create a new API key
- Copy the API key (starts with `AIzaSy...`)

### 2. **Set Environment Variable**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

**For permanent setup, add to your shell profile:**
```bash
echo 'export GEMINI_API_KEY="your_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Test TTS System**
```bash
python3 test_tts.py
```

### 5. **Run Full Demo**
```bash
python3 demo_tts.py
```

### 6. **Launch Streamlit App**
```bash
streamlit run streamlit_soda_advanced.py
```

## 🎯 **What You Get**

✅ **30 Different Robot Voices** - From "Puck -- Upbeat" to "Kore -- Firm"  
✅ **Emotional Speech Styles** - "cheerfully", "excitedly", "futuristically"  
✅ **Audio Caching** - Generated speech is saved for instant playback  
✅ **Custom TTS** - Make the robot say anything you want  
✅ **Voice Switching** - Change robot's personality on the fly  

## 🔧 **Troubleshooting**

### **"GEMINI_API_KEY not set"**
```bash
export GEMINI_API_KEY="your_key_here"
```

### **"google.genai import error"**
```bash
pip install google-genai==0.3.0
```

### **"protobuf version conflict"**
```bash
pip install protobuf==4.25.8
```

### **"No audio generated"**
- Check your internet connection
- Verify API key is correct
- Ensure you have sufficient API quota

## 📱 **Usage Examples**

### **Generate Robot Greeting**
```python
from tts_utils import RobotTTS
tts = RobotTTS()
audio_file = tts.get_robot_greeting_audio()
print(f"Audio saved to: {audio_file}")
```

### **Custom Speech**
```python
audio_file = tts.get_custom_audio(
    "Hello, I am your friendly robot assistant!", 
    "warmly"
)
```

### **Change Voice**
```python
tts.change_voice("Kore -- Firm")  # More serious tone
tts.change_voice("Puck -- Upbeat")  # Back to friendly
```

## 🌟 **Pro Tips**

1. **Pre-generate Audio** - Generate common phrases before your event
2. **Voice Selection** - Match voice personality to your robot's mood
3. **Style Instructions** - Use descriptive words like "enthusiastically", "softly"
4. **Audio Management** - Clear cache periodically to save disk space

## 🎉 **Ready to Go!**

Your SoDA robot now has a **real voice** that can:
- Greet visitors naturally
- Explain club information enthusiastically  
- Ask for gestures with excitement
- Say anything you want in any style

**The future of interactive robotics is here! 🤖🎤✨**
