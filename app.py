import os
import time
import tempfile
import threading
import streamlit as st
import base64
import io
import json
from datetime import datetime
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
from io import BytesIO
import requests
from gtts import gTTS
from groq import Groq
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.playback import play
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import uuid

# Page configuration
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2E5BFF;
        margin-bottom: 0;
        padding-bottom: 0;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #4F4F4F;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 300;
    }
    
    .status-listening {
        color: #4CAF50;
        font-weight: 600;
        padding: 8px;
        border-radius: 4px;
        background-color: rgba(76, 175, 80, 0.1);
        text-align: center;
    }
    
    .status-processing {
        color: #FFC107;
        font-weight: 600;
        padding: 8px;
        border-radius: 4px;
        background-color: rgba(255, 193, 7, 0.1);
        text-align: center;
    }
    
    .status-ready {
        color: #3F51B5;
        font-weight: 600;
        padding: 8px;
        border-radius: 4px;
        background-color: rgba(63, 81, 181, 0.1);
        text-align: center;
    }
    
    .status-error {
        color: #F44336;
        font-weight: 600;
        padding: 8px;
        border-radius: 4px;
        background-color: rgba(244, 67, 54, 0.1);
        text-align: center;
    }
    
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f8f9fa;
        padding: 1rem;
        text-align: center;
        border-top: 1px solid #e9ecef;
        font-size: 0.9rem;
        color: #6c757d;
        z-index: 1000;
    }
    
    .conversation-container {
        max-height: 55vh;
        overflow-y: auto;
        margin-bottom: 1rem;
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .message-timestamp {
        font-size: 0.7rem;
        color: #9e9e9e;
        margin-top: 4px;
    }
    
    .user-message {
        background-color: #E3F2FD;
        padding: 0.9rem;
        border-radius: 18px 18px 0 18px;
        margin-bottom: 1rem;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        position: relative;
    }
    
    .assistant-message {
        background-color: #F5F5F5;
        padding: 0.9rem;
        border-radius: 18px 18px 18px 0;
        margin-bottom: 1rem;
        width: fit-content;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        position: relative;
    }
    
    .controls-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .logo-container {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .logo {
        max-width: 220px;
        margin-bottom: 1rem;
    }
    
    .recording-status {
        padding: 0.8rem;
        margin: 0.8rem 0;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
    }
    
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2E5BFF;
        margin-bottom: 1.5rem;
    }
    
    .sidebar-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    .sidebar-section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #4F4F4F;
        margin-bottom: 0.5rem;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px;
    }
    
    .live-transcription {
        background-color: rgba(46, 91, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid #2E5BFF;
        font-style: italic;
    }
    
    .transcription-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #2E5BFF;
        margin-bottom: 0.5rem;
    }
    
    /* Adjust the main content padding to avoid footer overlap */
    .main {
        padding-bottom: 70px;
    }
    
    /* Pulsing animation for recording */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .recording-pulse {
        animation: pulse 1.5s infinite;
        background-color: rgba(244, 67, 54, 0.1);
        color: #F44336;
    }
    
    /* Language selector styling */
    .language-flag {
        width: 24px;
        height: 16px;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar styling */
    .conversation-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .conversation-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .conversation-container::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    
    .conversation-container::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        border-radius: 30px;
        margin-top: 8px;
    }
    
    /* Make sure the conversation container scrolls to bottom */
    .auto-scroll {
        overflow-y: auto;
        max-height: 55vh;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'language' not in st.session_state:
    st.session_state.language = "en"
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'recording_state' not in st.session_state:
    st.session_state.recording_state = 'stopped'
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'live_transcription' not in st.session_state:
    st.session_state.live_transcription = ""
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Language data with flags
languages = {
    "English": {"code": "en", "flag": "🇺🇸"},
    "Spanish": {"code": "es", "flag": "🇪🇸"},
    "French": {"code": "fr", "flag": "🇫🇷"},
    "German": {"code": "de", "flag": "🇩🇪"},
    "Italian": {"code": "it", "flag": "🇮🇹"},
    "Portuguese": {"code": "pt", "flag": "🇵🇹"},
    "Russian": {"code": "ru", "flag": "🇷🇺"},
    "Japanese": {"code": "ja", "flag": "🇯🇵"},
    "Chinese": {"code": "zh-CN", "flag": "🇨🇳"},
    "Hindi": {"code": "hi", "flag": "🇮🇳"},
    "Arabic": {"code": "ar", "flag": "🇸🇦"},
    "Korean": {"code": "ko", "flag": "🇰🇷"},
    "Dutch": {"code": "nl", "flag": "🇳🇱"},
    "Swedish": {"code": "sv", "flag": "🇸🇪"},
    "Turkish": {"code": "tr", "flag": "🇹🇷"}
}

# Function to create a unique audio filename
def get_audio_filename():
    return f"audio_{st.session_state.session_id}_{int(time.time())}.wav"

# Function to init Groq client
def init_groq_client(api_key):
    return Groq(api_key=api_key)

# Function to transcribe audio
def transcribe_with_groq(audio_path, api_key, language="en"):
    try:
        client = init_groq_client(api_key)
        with open(audio_path, "rb") as file:
            response = client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3-turbo",
                language=language
            )
        
        transcription = response.text.strip()
        return transcription
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return ""

# Function to get AI response
def get_llama_response(question, api_key, language="en"):
    try:
        client = init_groq_client(api_key)
        
        # Create system prompt based on selected language
        language_name = next((name for name, data in languages.items() if data["code"] == language), "English")
        
        system_prompt = f"""You are a helpful, friendly voice assistant. Keep your responses concise and conversational. 
        You're currently speaking in {language_name}. Respond in the same language as the user's question.
        Remember to be natural and friendly in your tone."""
        
        # Build messages with conversation history for context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for exchange in st.session_state.conversation_history:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Using Llama 3.3 70B model
            messages=messages,
            temperature=0.7,
            max_tokens=150,  # Keep responses short for voice
            top_p=1,
            stream=False
        )
        
        answer = completion.choices[0].message.content.strip()
        
        # Update conversation history
        st.session_state.conversation_history.append({"user": question, "assistant": answer})
        if len(st.session_state.conversation_history) > 5:  # Keep last 5 exchanges
            st.session_state.conversation_history.pop(0)
            
        return answer
    except Exception as e:
        st.error(f"Error calling Groq API: {str(e)}")
        return "I encountered an error while responding."

# Function to translate text
def translate_text(text, target_language="en"):
    try:
        translator = GoogleTranslator(source='auto', target=target_language)
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        st.error(f"Error in translation: {str(e)}")
        return text

# Function to save audio to file
def save_audio_to_file(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        return tmp_file.name

# Function to convert text to speech
def text_to_speech(text, language="en"):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts_filename = tmp_file.name
        
        # Generate speech
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(tts_filename)
        
        # Read the file and get the bytes
        with open(tts_filename, "rb") as f:
            audio_bytes = f.read()
        
        # Clean up
        if os.path.exists(tts_filename):
            os.remove(tts_filename)
        
        return audio_bytes
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")
        return None

# Function to process audio
def process_audio(audio_bytes, api_key):
    st.session_state.is_processing = True
    
    try:
        # Save audio to file
        audio_file = save_audio_to_file(audio_bytes)
        
        # Transcribe audio
        transcription = transcribe_with_groq(audio_file, api_key, st.session_state.language)
        
        if not transcription:
            st.warning("No speech detected or transcription failed")
            return None, None
        
        # Get AI response
        response = get_llama_response(transcription, api_key, st.session_state.language)
        
        # Generate audio response
        audio_response = text_to_speech(response, st.session_state.language)
        
        # Add to conversation with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.conversation.append({
            "user": transcription, 
            "assistant": response,
            "timestamp": timestamp
        })
        
        # Clean up audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        return transcription, audio_response
    
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None
    
    finally:
        st.session_state.is_processing = False

# Function to create downloadable link for conversation
def get_conversation_download_link():
    # Create markdown content
    conversation_md = "# Voice Assistant Conversation\n\n"
    conversation_md += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Get language name
    language_name = next((name for name, data in languages.items() if data["code"] == st.session_state.language), "English")
    conversation_md += f"Language: {language_name}\n\n"
    
    # Add conversation entries
    for entry in st.session_state.conversation:
        time_str = entry.get("timestamp", "")
        conversation_md += f"## User [{time_str}]\n{entry['user']}\n\n"
        conversation_md += f"## Assistant [{time_str}]\n{entry['assistant']}\n\n"
        conversation_md += "---\n\n"
    
    # Convert to bytes for download
    b64 = base64.b64encode(conversation_md.encode()).decode()
    filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}" class="download-btn">Download Conversation</a>'
    return href

# Function to get the data URL for the audio as a downloadable link
def get_audio_download_link(audio_bytes, filename):
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download Audio</a>'
    return href

# Class for audio processing
class AudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.should_stop = False
        self.audio_thread = None
        self.frames = []
        self.sample_rate = 16000
    
    def start_recording(self):
        self.is_recording = True
        self.should_stop = False
        self.frames = []
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def stop_recording(self):
        self.should_stop = True
        if self.audio_thread:
            self.audio_thread.join(timeout=1)
        self.is_recording = False
        
        # Return audio bytes if we have any frames
        if self.frames:
            return self._get_audio_bytes()
        return None
    
    def _record_audio(self):
        # Record audio in chunks
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            self.frames.append(indata.copy())
        
        with sd.InputStream(callback=callback, channels=1, samplerate=self.sample_rate):
            while not self.should_stop:
                sd.sleep(100)  # Sleep for 100ms
    
    def _get_audio_bytes(self):
        if not self.frames:
            return None
        
        # Combine all frames
        audio_data = np.concatenate(self.frames, axis=0)
        
        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, self.sample_rate, format='WAV')
        buffer.seek(0)
        return buffer.read()

# Initialize audio processor
audio_processor = AudioProcessor()

# Header with logo
st.markdown("""
<div class="logo-container">
    <img src="https://s3-eu-west-1.amazonaws.com/tpd/logos/60d3a0bc65022800013b18b3/0x0.png" class="logo" alt="AI Voice Assistant Logo">
    <h1 class="main-header">AI Voice Assistant</h1>
    <p class="sub-header">Your AI-powered multilingual voice companion</p>
</div>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([3, 1])

with col1:
    # Conversation display
    st.markdown("<h3>Conversation</h3>", unsafe_allow_html=True)
    
    conversation_container = st.container()
    
    with conversation_container:
        st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
        
        # Display conversation with timestamps
        for entry in st.session_state.conversation:
            time_str = entry.get("timestamp", "")
            st.markdown(f'<div class="user-message">{entry["user"]}<div class="message-timestamp">{time_str}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message">{entry["assistant"]}<div class="message-timestamp">{time_str}</div></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Live transcription display
    if st.session_state.recording_state == 'recording':
        st.markdown("""
        <div class="live-transcription">
            <div class="transcription-title">Live Transcription</div>
            <div id="transcription-content">Listening...</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Sidebar for configuration
    st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)
    
    # API Key input
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">API Key</div>', unsafe_allow_html=True)
    api_key = st.text_input("Groq API Key", value="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", type="password")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Language selection
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Language</div>', unsafe_allow_html=True)
    
    language_options = [f"{data['flag']} {name}" for name, data in languages.items()]
    selected_language = st.selectbox("Select Language", language_options, index=0)
    
    # Extract language code from selection
    selected_language_name = selected_language.split(" ", 1)[1]
    st.session_state.language = languages[selected_language_name]["code"]
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recording controls
    st.markdown('<div class="controls-container">', unsafe_allow_html=True)
    
    status_container = st.empty()
    
    if st.session_state.recording_state == 'recording':
        status_container.markdown('<p class="status-listening recording-pulse">Recording... 🎙️</p>', unsafe_allow_html=True)
    elif st.session_state.is_processing:
        status_container.markdown('<p class="status-processing">Processing... ⏳</p>', unsafe_allow_html=True)
    else:
        status_container.markdown('<p class="status-ready">Ready 🎤</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎙️ Start", 
                   type="primary" if st.session_state.recording_state != 'recording' else "secondary",
                   disabled=st.session_state.recording_state == 'recording'):
            st.session_state.recording_state = 'recording'
            audio_processor.start_recording()
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop", 
                   type="primary" if st.session_state.recording_state == 'recording' else "secondary",
                   disabled=st.session_state.recording_state != 'recording'):
            st.session_state.recording_state = 'stopped'
            audio_bytes = audio_processor.stop_recording()
            if audio_bytes:
                st.session_state.audio_bytes = audio_bytes
            st.rerun()
    
    if st.button("🔄 Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.conversation_history = []
        st.rerun()
    
    if st.session_state.conversation:
        st.markdown(get_conversation_download_link(), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Process recorded audio
if st.session_state.audio_bytes and not st.session_state.is_processing:
    # Display the recorded audio
    st.sidebar.audio(st.session_state.audio_bytes, format="audio/wav")
    
    # Process the audio
    with st.spinner("Processing your message..."):
        transcription, audio_response = process_audio(st.session_state.audio_bytes, api_key)
        
        if transcription and audio_response:
            # Play the audio response
            st.sidebar.audio(audio_response, format="audio/mp3")
    
    # Clear the audio bytes to prevent reprocessing
    st.session_state.audio_bytes = None
    
    # Rerun to update the UI
    st.rerun()

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 AI Voice Assistant | Powered by Groq, Whisper, and Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Auto-scroll to bottom of conversation
if st.session_state.conversation:
    st.markdown("""
    <script>
        var element = document.querySelector('.conversation-container');
        element.scrollTop = element.scrollHeight;
    </script>
    """, unsafe_allow_html=True)
