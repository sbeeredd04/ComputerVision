# 🤖 SoDA Club Fair Attraction

An interactive Streamlit application that combines gesture recognition with an animated robot to create an engaging club fair experience. Now featuring **Text-to-Speech (TTS)** powered by Google's Gemini API!

## ✨ Features

- **Split-screen Interface**: Live camera feed on the left, animated robot on the right
- **Real-time Gesture Recognition**: Detects thumbs up, heart, and peace gestures
- **Person Detection**: Automatically detects when someone is in front of the camera
- **Interactive Robot**: Animated robot that responds to gestures and speaks
- **🎤 Text-to-Speech**: Robot speaks with natural, customizable voices using Gemini TTS
- **Conversation Flow**: Automated conversation pipeline that guides users through the experience
- **QR Code Generation**: Displays QR code for club signup when heart gesture is detected
- **Voice Customization**: Choose from 30 different voice personalities

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam access
- Modern web browser
- **Google Gemini API Key** (for TTS functionality)

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   cd notebooks/tutorials
   ```

2. **Set up your Gemini API key**:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
   
   Or create a `.env` file:
   ```bash
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run streamlit_soda_advanced.py
   ```

5. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## 🎤 Text-to-Speech Features

### Voice Generation
- **Automatic Speech**: Robot automatically generates speech for all interactions
- **30 Voice Options**: Choose from voices like "Puck -- Upbeat", "Kore -- Firm", "Fenrir -- Excitable"
- **Style Control**: Add style instructions like "cheerfully", "excitedly", "softly"
- **Audio Caching**: Generated audio is cached for instant playback

### TTS Controls
- **Generate Greeting Audio**: Create speech for robot greetings
- **Generate SoDA Audio**: Create speech for club information
- **Generate Heart Audio**: Create speech for heart gesture requests
- **Custom TTS**: Generate speech for any custom text
- **Voice Selection**: Change robot's voice personality
- **Audio Management**: Play all generated audio files

## 🎯 How It Works

### Automatic Flow

1. **Person Detection**: Camera detects when someone is in view for at least 1 second
2. **Greeting**: Robot automatically greets the person and asks for a thumbs up
3. **Thumbs Up Gesture**: When detected, robot talks about SoDA club
4. **Heart Gesture**: Robot asks for a heart gesture to show interest
5. **QR Code**: Displays QR code linking to the club's website

### Manual Controls

- **👋 Greet**: Manually trigger robot greeting
- **💬 Talk About SoDA**: Manually trigger SoDA information
- **❤️ Ask for Heart**: Manually ask for heart gesture
- **🔄 Reset**: Reset all conversation states

## 🎮 Gesture Recognition

The application recognizes several hand gestures:

- **Thumbs Up**: Thumb extended upward, other fingers closed
- **Heart**: Index and middle fingers extended and curved downward
- **Peace**: Index and middle fingers extended, ring and pinky closed

## 🛠️ Technical Details

### Architecture

- **Frontend**: Streamlit with HTML/CSS/JavaScript components
- **Computer Vision**: OpenCV + MediaPipe for hand tracking
- **Object Detection**: YOLO for person detection
- **Real-time Processing**: WebRTC for live video streaming
- **Animation**: CSS animations and JavaScript for robot movements
- **Text-to-Speech**: Google Gemini API for natural voice generation

### Key Components

- `GestureDetector`: Video transformer class for real-time gesture recognition
- `SoDARobot`: JavaScript class managing robot animations and speech
- `RobotTTS`: Python class for Gemini TTS integration
- `streamlit_soda_advanced.py`: Main Streamlit application

## 🔧 Configuration

### Environment Variables

- **GEMINI_API_KEY**: Required for TTS functionality
- **SIGNUP_URL**: Update to point to your actual club signup page

### Customization

1. **Club Information**: Update `CLUB_NAME` and `SIGNUP_URL` in the Python files
2. **Robot Appearance**: Modify CSS in the `get_robot_html()` function
3. **Gesture Sensitivity**: Adjust confidence thresholds in gesture detection
4. **Voice Selection**: Choose from 30 different voice personalities
5. **Timing**: Modify delays and durations for different conversation stages

## 📱 Usage Instructions

### For Users

1. **Start the camera**: Click "Start" in the camera section
2. **Show your face**: Stand in front of the camera
3. **Follow prompts**: The robot will guide you through the experience
4. **Use gestures**: Show thumbs up and heart gestures when prompted
5. **Scan QR code**: Use your phone to scan the displayed QR code

### For Demonstrators

1. **Manual controls**: Use the control buttons for demonstrations
2. **TTS controls**: Generate and customize robot speech
3. **Voice selection**: Change robot's voice personality
4. **Reset functionality**: Use reset button to start over
5. **State monitoring**: Watch the current state indicators

## 🐛 Troubleshooting

### Common Issues

1. **Camera not working**:
   - Check browser permissions
   - Ensure no other applications are using the camera
   - Try refreshing the page

2. **Gestures not detected**:
   - Ensure good lighting
   - Keep hands clearly visible
   - Check gesture confidence thresholds

3. **TTS not working**:
   - Verify GEMINI_API_KEY is set correctly
   - Check internet connection
   - Ensure google-genai package is installed

4. **Robot not responding**:
   - Check browser console for JavaScript errors
   - Ensure HTML components are loading properly
   - Try manual control buttons

### Performance Tips

- Use a modern browser (Chrome, Firefox, Safari)
- Ensure good lighting for gesture recognition
- Close other resource-intensive applications
- Use a computer with decent processing power
- Generate TTS audio in advance for smoother experience

## 🎨 Customization

### Robot Appearance

The robot's appearance can be customized by modifying the CSS in the `get_robot_html()` function:

- Colors and gradients
- Sizes and proportions
- Animation timings
- Speech bubble styling

### Voice Customization

Modify the robot's voice using the TTS controls:

- Choose from 30 different voice personalities
- Add style instructions for emotional expression
- Generate custom speech for any text
- Cache audio for instant playback

### Conversation Flow

Modify the conversation logic in the `GestureDetector.transform()` method:

- Add new gesture types
- Change conversation states
- Modify timing and responses
- Add new robot actions

## 📚 Dependencies

- **streamlit**: Web application framework
- **opencv-python**: Computer vision library
- **mediapipe**: Hand tracking and gesture recognition
- **ultralytics**: YOLO object detection
- **streamlit-webrtc**: Real-time video streaming
- **qrcode**: QR code generation
- **Pillow**: Image processing
- **google-genai**: Google Gemini API integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- MediaPipe team for hand tracking capabilities
- Ultralytics for YOLO implementation
- Streamlit team for the web framework
- OpenCV community for computer vision tools
- Google Gemini team for TTS capabilities

## 📞 Support

For questions or issues:

1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub
4. Contact the development team

---

**Happy coding! 🤖✨**
