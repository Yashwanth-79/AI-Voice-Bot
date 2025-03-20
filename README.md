# AI Voice Assistant

Live : https://home-llc-ai-voice-bot.streamlit.app/

## Summary
The AI Voice Assistant is an interactive, multi-language voice-based chatbot that utilizes Groq's AI models for speech recognition, transcription, and response generation. It converts user speech into text, processes it using an LLM, and provides AI-generated responses in both text and audio formats.

## Features
- 🎙️ **Voice Input & Output:** Users can interact using their voice, and responses are spoken back.
- 🌍 **Multi-Language Support:** Supports multiple languages including English, Spanish, French, German, and more.
- 🤖 **AI-Powered Responses:** Uses `Groq` AI models for intelligent responses.
- 📝 **Real-Time Transcription:** Converts speech to text using Whisper.
- 🎨 **Custom UI Design:** Styled using Streamlit's custom CSS and layout options.
- 🔐 **API Key Authentication:** Requires Groq API Key for functioning.

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Groq API Key (Get it from [Groq Console](https://console.groq.com/playground))
- Required Libraries:
  ```bash
  pip install streamlit gtts groq audio_recorder_streamlit
  ```

### Running the Application
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-voice-assistant.git
   cd ai-voice-assistant
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Enter your **Groq API Key** when prompted in the UI.

---

## Code Structure

### 1. `app.py` - Main Application File
This file contains the core logic of the AI Voice Assistant:
- **UI Elements:** Streamlit components for user interaction.
- **Session Management:** Maintains conversation history.
- **Voice Recording:** Uses `audio_recorder_streamlit` for audio capture.
- **API Calls:** Sends recorded speech to Groq API for processing.

## Design Decisions

### 1. **Streamlit for UI**
- Chosen for rapid development and interactivity.
- Supports real-time updates and user-friendly layouts.

### 2. **Groq for AI Processing**
- Utilized Whisper for speech-to-text conversion.
- Uses `llama-3.3-70b-versatile` for chatbot responses.

### 3. **gTTS for Text-to-Speech**
- Converts AI responses into spoken audio.
- Allows multilingual speech output.

### 4. **Custom Styling**
- CSS is used to enhance UI elements.
- Scrollable conversation view.

---

## Future Improvements
- **Support for More Languages**
- **Improved AI Response Quality** using better prompts
- **Mobile Compatibility Enhancements**

---

## Contributing
Pull requests are welcome! Please ensure changes are well-tested.

---

