import os
import time
import tempfile
import threading
import streamlit as st
import audio_recorder_streamlit as ast
from gtts import gTTS
from datetime import datetime
from groq import Groq
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import wave
from io import BytesIO

# ------------------ PAGE CONFIGURATION ------------------
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎙️",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E5BFF;
        margin-bottom: 0;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #4F4F4F;
        margin-top: 0;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 300;
    }
    
    .conversation-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 10px;
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #E3F2FD;
        padding: 0.8rem;
        border-radius: 15px 15px 0 15px;
        margin-bottom: 1rem;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
    }
    
    .assistant-message {
        background-color: #F5F5F5;
        padding: 0.8rem;
        border-radius: 15px 15px 15px 0;
        margin-bottom: 1rem;
        width: fit-content;
        max-width: 80%;
    }
    
    .message-timestamp {
        font-size: 0.7rem;
        color: #9e9e9e;
        margin-top: 4px;
    }
    
    .live-transcription {
        background-color: #E8F5E9;
        padding: 0.8rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
        font-style: italic;
        border: 1px dashed #81C784;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar styling */
    .conversation-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .conversation-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .conversation-container::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2E5BFF;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1E3FB3;
        transform: translateY(-2px);
    }
    
    /* Listening indicator */
    .listening-indicator {
        display: inline-block;
        padding: 5px 15px;
        background-color: #ef5350;
        color: white;
        border-radius: 20px;
        font-weight: 500;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
        100% {
            opacity: 1;
        }
    }
</style>
""", unsafe_allow_html=True)

# ------------------ INITIALIZE SESSION STATE ------------------
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'listening' not in st.session_state:
    st.session_state.listening = False
if 'live_transcription' not in st.session_state:
    st.session_state.live_transcription = ""
if 'language' not in st.session_state:
    st.session_state.language = "en"
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'recording_thread' not in st.session_state:
    st.session_state.recording_thread = None
if 'transcription_thread' not in st.session_state:
    st.session_state.transcription_thread = None
if 'stop_recording' not in st.session_state:
    st.session_state.stop_recording = threading.Event()
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'current_audio_file' not in st.session_state:
    st.session_state.current_audio_file = ""
if 'auto_play_response' not in st.session_state:
    st.session_state.auto_play_response = True

# ------------------ LANGUAGE OPTIONS ------------------
languages = {
    "English": {"code": "en", "flag": "🇺🇸"},
    "Spanish": {"code": "es", "flag": "🇪🇸"},
    "French": {"code": "fr", "flag": "🇫🇷"},
    "German": {"code": "de", "flag": "🇩🇪"},
    "Japanese": {"code": "ja", "flag": "🇯🇵"},
    "Chinese": {"code": "zh-CN", "flag": "🇨🇳"},
    "Hindi": {"code": "hi", "flag": "🇮🇳"},
    "Arabic": {"code": "ar", "flag": "🇸🇦"}
}

# ------------------ FUNCTIONS ------------------
def init_groq_client(api_key):
    """Initialize Groq client with API key"""
    return Groq(api_key=api_key)

def audio_callback(indata, frames, time, status):
    """Callback function for the audio stream"""
    if status:
        print(f"Stream status: {status}")
    st.session_state.audio_queue.put(indata.copy())

def save_audio_stream(filename, sample_rate=16000, channels=1):
    """Save audio stream to a file"""
    try:
        # Open the audio stream
        with sd.InputStream(callback=audio_callback, channels=channels, samplerate=sample_rate):
            # Create a wave file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                
                # Continue saving audio until stop_recording event is set
                while not st.session_state.stop_recording.is_set():
                    if not st.session_state.audio_queue.empty():
                        chunk = st.session_state.audio_queue.get()
                        wf.writeframes((chunk * 32767).astype(np.int16).tobytes())
                    else:
                        time.sleep(0.01)
    except Exception as e:
        print(f"Error in save_audio_stream: {e}")

def transcribe_with_groq(audio_path, api_key, language="en"):
    """Transcribe audio using Groq's Whisper API"""
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

def get_llama_response(question, api_key, language="en"):
    """Get response from Groq's Llama model with conversation history"""
    try:
        client = init_groq_client(api_key)
        
        # Get language name for system prompt
        language_name = next((name for name, data in languages.items() if data["code"] == language), "English")
        
        # Create system prompt
        system_prompt = f"""You are a helpful, friendly voice assistant. Keep your responses concise and conversational. 
        You're currently speaking in {language_name}. Respond in the same language as the user's question."""
        
        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for context
        for exchange in st.session_state.conversation_history:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Get completion from Groq
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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

def text_to_speech(text, language="en"):
    """Convert text to speech and return audio bytes"""
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

def start_recording(api_key):
    """Start recording and transcribing audio"""
    if st.session_state.listening or not api_key:
        return
    
    # Set listening state
    st.session_state.listening = True
    st.session_state.live_transcription = "Listening..."
    st.session_state.stop_recording.clear()
    
    # Create temp file for audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        audio_file = tmp_file.name
    
    # Start recording thread
    recording_thread = threading.Thread(
        target=save_audio_stream, 
        args=(audio_file,)
    )
    recording_thread.daemon = True
    recording_thread.start()
    st.session_state.recording_thread = recording_thread
    
    # Start real-time transcription thread
    transcription_thread = threading.Thread(
        target=real_time_transcription,
        args=(audio_file, api_key)
    )
    transcription_thread.daemon = True
    transcription_thread.start()
    st.session_state.transcription_thread = transcription_thread
    
    # Store the audio file path
    st.session_state.current_audio_file = audio_file
    
    st.rerun()

def stop_recording():
    """Stop recording and process the final audio"""
    if not st.session_state.listening:
        return
    
    # Signal the recording to stop
    st.session_state.stop_recording.set()
    time.sleep(0.5)  # Give time for threads to clean up
    
    # Set state
    st.session_state.listening = False
    st.session_state.is_processing = True
    
    # Wait for threads to finish
    if st.session_state.recording_thread:
        st.session_state.recording_thread.join(timeout=1)
    if st.session_state.transcription_thread:
        st.session_state.transcription_thread.join(timeout=1)
    
    # Process the final audio
    audio_file = st.session_state.current_audio_file
    if os.path.exists(audio_file):
        # Process the full audio
        process_final_audio(audio_file, st.session_state.api_key)
        
    # Reset state
    st.session_state.live_transcription = ""
    st.session_state.is_processing = False
    
    st.rerun()

def real_time_transcription(audio_file, api_key):
    """Perform real-time transcription on audio segments"""
    try:
        # Wait for some audio to be recorded
        time.sleep(2)
        
        last_transcription_time = time.time()
        
        while not st.session_state.stop_recording.is_set():
            current_time = time.time()
            
            # Only transcribe every few seconds to avoid overwhelming the API
            if current_time - last_transcription_time > 3:
                # Create a temporary copy of the current audio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_segment:
                    segment_file = tmp_segment.name
                
                try:
                    # Copy the current audio data
                    with open(audio_file, 'rb') as src:
                        with open(segment_file, 'wb') as dst:
                            dst.write(src.read())
                    
                    # Transcribe the segment
                    partial_transcription = transcribe_with_groq(
                        segment_file, 
                        api_key, 
                        st.session_state.language
                    )
                    
                    if partial_transcription:
                        st.session_state.live_transcription = partial_transcription
                except Exception as e:
                    print(f"Error in segment transcription: {e}")
                finally:
                    # Clean up segment file
                    if os.path.exists(segment_file):
                        os.remove(segment_file)
                
                last_transcription_time = current_time
            
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error in real_time_transcription: {e}")

def process_final_audio(audio_file, api_key):
    """Process the final audio recording"""
    try:
        # Final transcription
        transcription = transcribe_with_groq(audio_file, api_key, st.session_state.language)
        
        if not transcription:
            st.warning("No speech detected or transcription failed")
            return
        
        # Get AI response
        response = get_llama_response(transcription, api_key, st.session_state.language)
        
        # Generate audio response
        audio_response = text_to_speech(response, st.session_state.language)
        
        # Add to conversation with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.conversation.append({
            "user": transcription, 
            "assistant": response,
            "timestamp": timestamp,
            "audio_response": audio_response,
            "auto_played": False  # Track whether this response has been auto-played
        })
        
        # Clean up audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
    except Exception as e:
        st.error(f"Error processing final audio: {str(e)}")

# ------------------ APP LAYOUT ------------------
# Header
st.markdown("<h1 class='main-header'>AI Voice Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Your AI-powered multilingual voice companion</p>", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([3, 1])

with col1:
    # Conversation display container
    st.subheader("Conversation")
    
    # Create a container for the conversation
    conversation_container = st.container()
    
    with conversation_container:
        st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
        
        # Display the conversation history
        for i, entry in enumerate(st.session_state.conversation):
            time_str = entry.get("timestamp", "")
            st.markdown(f'<div class="user-message">{entry["user"]}<div class="message-timestamp">{time_str}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message">{entry["assistant"]}<div class="message-timestamp">{time_str}</div></div>', unsafe_allow_html=True)
            
            # Play the audio response if available and auto-play is enabled
            if "audio_response" in entry and entry["audio_response"]:
                st.audio(entry["audio_response"], format="audio/mp3", start_time=0 if st.session_state.auto_play_response and not entry.get("auto_played", True) else None)
                # Mark this response as auto-played
                if st.session_state.auto_play_response and not entry.get("auto_played", True):
                    st.session_state.conversation[i]["auto_played"] = True
        
        # Display live transcription if listening
        if st.session_state.listening and st.session_state.live_transcription:
            st.markdown(f'<div class="live-transcription">{st.session_state.live_transcription}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Settings and controls
    st.subheader("Controls")
    
    # API Key input
    api_key = st.text_input("Groq API Key", value="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", type="password")
    st.session_state.api_key = api_key
    
    # Language selection
    language_options = [f"{data['flag']} {name}" for name, data in languages.items()]
    selected_language = st.selectbox("Select Language", language_options, index=0)
    
    # Extract language code from selection
    selected_language_name = selected_language.split(" ", 1)[1]
    st.session_state.language = languages[selected_language_name]["code"]
    
    # Auto-play option
    st.session_state.auto_play_response = st.checkbox("Auto-play responses", value=True)
    
    # Status indicator
    if st.session_state.listening:
        st.markdown('<div class="listening-indicator">Listening...</div>', unsafe_allow_html=True)
    elif st.session_state.is_processing:
        st.info("Processing... ⏳")
    else:
        st.success("Ready 🎤")
    
    # Voice recording control buttons
    col_start, col_stop = st.columns(2)
    
    with col_start:
        if st.button("🎤 Start Talking", disabled=st.session_state.listening or st.session_state.is_processing):
            if api_key:
                start_recording(api_key)
            else:
                st.error("Please enter a valid API key")
    
    with col_stop:
        if st.button("🛑 Stop", disabled=not st.session_state.listening):
            stop_recording()
    
    # Microphone access using audio_recorder_streamlit for initial setup only
    st.markdown("#### Or record a single message:")
    audio_bytes = ast.audio_recorder(
        text="",
        recording_color="#e53935",
        neutral_color="#2E5BFF",
        icon_size="2x"
    )
    
    # Clear conversation button
    if st.button("🔄 Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.conversation_history = []
        st.session_state.live_transcription = ""
        st.rerun()

# Process recorded audio from audio_recorder if used
if audio_bytes and not st.session_state.listening and not st.session_state.is_processing:
    # Set processing state
    st.session_state.is_processing = True
    st.rerun()  # Update UI to show processing state
    
    # Save audio to file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        audio_file = tmp_file.name
    
    # Process the audio
    # Transcribe audio
    transcription = transcribe_with_groq(audio_file, api_key, st.session_state.language)
    
    if transcription:
        # Get AI response
        response = get_llama_response(transcription, api_key, st.session_state.language)
        
        # Generate audio response
        audio_response = text_to_speech(response, st.session_state.language)
        
        # Add to conversation
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.conversation.append({
            "user": transcription, 
            "assistant": response,
            "timestamp": timestamp,
            "audio_response": audio_response,
            "auto_played": False  # New responses haven't been auto-played yet
        })
    
    # Clean up audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)
    
    # Reset processing state
    st.session_state.is_processing = False
    st.rerun()  # Update UI with new conversation
