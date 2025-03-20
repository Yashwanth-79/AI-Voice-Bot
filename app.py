import os
import time
import tempfile
import threading
import streamlit as st
from gtts import gTTS
from datetime import datetime
from groq import Groq
import sounddevice as sd
import numpy as np
import queue
import wave
import pyaudio  # Using pyaudio for more reliable recording

# ------------------ PAGE CONFIGURATION ------------------
st.set_page_config(page_title="AI Voice Assistant", page_icon="🎙️", layout="wide")

# ------------------ CUSTOM CSS ------------------ (No changes needed here)
st.markdown("""
<style>
    /* ... (Your existing CSS, no changes needed) ... */
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
if 'playback_thread' not in st.session_state:
    st.session_state.playback_thread = None
if 'audio_data' not in st.session_state:  # For single message processing
    st.session_state.audio_data = None

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

# ------------------ AUDIO CONFIGURATION ------------------
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# ------------------ FUNCTIONS ------------------
def init_groq_client(api_key):
    return Groq(api_key=api_key)

def record_audio(audio_file):
    """Records audio using PyAudio and saves to a file."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    while not st.session_state.stop_recording.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)  # Handle overflow
            frames.append(data)
        except IOError as e:
            if e.errno == pyaudio.paInputOverflowed:
                print("Input overflowed, continuing...")
                continue  # Continue to the next iteration
            else:
                raise

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(audio_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print(f"Audio saved to {audio_file}")

def transcribe_with_groq(audio_path, api_key, language="en"):
    try:
        client = init_groq_client(api_key)
        with open(audio_path, "rb") as file:
            response = client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3-turbo",
                language=language
            )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return ""

def get_llama_response(question, api_key, language="en"):
    try:
        client = init_groq_client(api_key)
        language_name = next((name for name, data in languages.items() if data["code"] == language), "English")
        system_prompt = (f"You are a helpful, friendly voice assistant. Keep your responses concise and conversational. "
                         f"You're currently speaking in {language_name}. Respond in the same language as the user's question.")
        messages = [{"role": "system", "content": system_prompt}]
        for exchange in st.session_state.conversation_history:
            messages.extend([
                {"role": "user", "content": exchange["user"]},
                {"role": "assistant", "content": exchange["assistant"]}
            ])
        messages.append({"role": "user", "content": question})
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=150,
            top_p=1,
            stream=False
        )
        answer = completion.choices[0].message.content.strip()
        st.session_state.conversation_history.append({"user": question, "assistant": answer})
        if len(st.session_state.conversation_history) > 5:
            st.session_state.conversation_history.pop(0)
        return answer
    except Exception as e:
        st.error(f"Error calling Groq API: {str(e)}")
        return "I encountered an error while responding."

def text_to_speech(text, language="en"):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(tmp_file.name)
            with open(tmp_file.name, "rb") as f:
                audio_bytes = f.read()
        os.remove(tmp_file.name)
        return audio_bytes
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")
        return None

def start_recording(api_key):
    if st.session_state.listening:
        return

    st.session_state.listening = True
    st.session_state.live_transcription = "Listening..."
    st.session_state.stop_recording.clear()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        audio_file = tmp_file.name

    recording_thread = threading.Thread(target=record_audio, args=(audio_file,))
    recording_thread.daemon = True
    recording_thread.start()
    st.session_state.recording_thread = recording_thread

    transcription_thread = threading.Thread(target=real_time_transcription, args=(audio_file, api_key))
    transcription_thread.daemon = True
    transcription_thread.start()
    st.session_state.transcription_thread = transcription_thread

    st.session_state.current_audio_file = audio_file
    st.rerun()

def stop_recording():
    if not st.session_state.listening:
        return

    st.session_state.stop_recording.set()
    time.sleep(0.5)  # Allow time for final writes

    st.session_state.listening = False

    if st.session_state.recording_thread:
        st.session_state.recording_thread.join(timeout=2)  # Increased timeout
    if st.session_state.transcription_thread:
        st.session_state.transcription_thread.join(timeout=2)

    audio_file = st.session_state.current_audio_file
    if os.path.exists(audio_file):
        process_final_audio(audio_file, st.session_state.api_key)

    st.session_state.live_transcription = ""
    st.rerun()

def real_time_transcription(audio_file, api_key):
    try:
        time.sleep(2)  # Wait for some audio
        last_transcription_time = time.time()

        while not st.session_state.stop_recording.is_set():
            current_time = time.time()
            if current_time - last_transcription_time > 3:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_segment:
                    segment_file = tmp_segment.name
                try:
                    with open(audio_file, 'rb') as src, open(segment_file, 'wb') as dst:
                        dst.write(src.read())
                    partial_transcription = transcribe_with_groq(segment_file, api_key, st.session_state.language)
                    if partial_transcription:
                        st.session_state.live_transcription = partial_transcription
                except Exception as e:
                    print(f"Error in segment transcription: {e}")
                finally:
                    os.remove(segment_file)
                last_transcription_time = current_time
            time.sleep(0.1)
    except Exception as e:
        print(f"Error in real_time_transcription: {e}")

def process_final_audio(audio_file, api_key):
    try:
        transcription = transcribe_with_groq(audio_file, api_key, st.session_state.language)
        if not transcription:
            st.warning("No speech detected or transcription failed")
            return

        response = get_llama_response(transcription, api_key, st.session_state.language)
        audio_response = text_to_speech(response, st.session_state.language)

        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.conversation.append({
            "user": transcription,
            "assistant": response,
            "timestamp": timestamp,
            "audio_response": audio_response
        })
        if audio_response:
            st.session_state.audio_queue.put(audio_response)
            if st.session_state.playback_thread is None or not st.session_state.playback_thread.is_alive():
                st.session_state.playback_thread = threading.Thread(target=play_audio_responses, daemon=True)
                st.session_state.playback_thread.start()

        os.remove(audio_file)
    except Exception as e:
        st.error(f"Error processing final audio: {str(e)}")

def play_audio_responses():
    while True:
        try:
            audio_bytes = st.session_state.audio_queue.get(timeout=0.1)
            if audio_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(audio_bytes)
                    os.system(f"afplay {tmp_file.name}" if os.name == 'posix' else f"start /B {tmp_file.name}")  # Cross-platform
                os.remove(tmp_file.name)
        except queue.Empty:
            if st.session_state.stop_recording.is_set() and st.session_state.audio_queue.empty():
                break
            continue
        except Exception as e:
            st.error(f"Error playing audio: {e}")
            break

def process_single_message(audio_bytes, api_key):
    """Processes a single audio message (from audio_recorder_streamlit)."""
    if not audio_bytes:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        audio_file = tmp_file.name

    try:
        transcription = transcribe_with_groq(audio_file, api_key, st.session_state.language)
        if not transcription:
            st.warning("No speech detected or transcription failed")
            return

        response = get_llama_response(transcription, api_key, st.session_state.language)
        audio_response = text_to_speech(response, st.session_state.language)

        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.conversation.append({
            "user": transcription,
            "assistant": response,
            "timestamp": timestamp,
            "audio_response": audio_response
        })

        # Queue audio for playback
        if audio_response:
            st.session_state.audio_queue.put(audio_response)
            if st.session_state.playback_thread is None or not st.session_state.playback_thread.is_alive():
                st.session_state.playback_thread = threading.Thread(target=play_audio_responses, daemon=True)
                st.session_state.playback_thread.start()


    except Exception as e:
        st.error(f"Error processing single message: {e}")
    finally:
        os.remove(audio_file)  # Clean up

# ------------------ APP LAYOUT ------------------
st.markdown("<h1 class='main-header'>AI Voice Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Your AI-powered multilingual voice companion</p>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Conversation")
    conversation_container = st.container()
    with conversation_container:
        st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
        for entry in st.session_state.conversation:
            time_str = entry.get("timestamp", "")
            st.markdown(f'<div class="user-message">{entry["user"]}<div class="message-timestamp">{time_str}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message">{entry["assistant"]}<div class="message-timestamp">{time_str}</div></div>', unsafe_allow_html=True)
        if st.session_state.listening and st.session_state.live_transcription:
            st.markdown(f'<div class="live-transcription">{st.session_state.live_transcription}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("Controls")
    api_key = st.text_input("Groq API Key", type="password")
    st.session_state.api_key = api_key

    language_options = [f"{data['flag']} {name}" for name, data in languages.items()]
    selected_language = st.selectbox("Select Language", language_options, index=0)
    selected_language_name = selected_language.split(" ", 1)[1]
    st.session_state.language = languages[selected_language_name]["code"]

    if st.session_state.listening:
        st.markdown('<div class="listening-indicator">Listening...</div>', unsafe_allow_html=True)
    else:
        st.success("Ready 🎤")

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("🎤 Start Talking", disabled=st.session_state.listening):
            if api_key:
                start_recording(api_key)
            else:
                st.warning("Please enter your Groq API Key.")
    with col_stop:
        if st.button("🛑 Stop", disabled=not st.session_state.listening):
            stop_recording()

    st.markdown("#### Or record a single message:")
    audio_bytes = st.session_state.audio_data = st.session_state.audio_data = ast.audio_recorder(
        text="",
        recording_color="#e53935",
        neutral_color="#2E5BFF",
        icon_size="2x"
    )
    if audio_bytes and st.session_state.audio_data != audio_bytes:
        st.session_state.audio_data = audio_bytes
        process_single_message(audio_bytes, api_key)
        st.rerun()

    if st.button("🔄 Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.conversation_history = []
        st.session_state.live_transcription = ""
        st.rerun()
