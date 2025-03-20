import os
import time
import queue
import tempfile
import threading
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
from groq import Groq
import base64
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 2rem;
    }
    .status-listening {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-processing {
        color: #FFC107;
        font-weight: bold;
    }
    .status-error {
        color: #F44336;
        font-weight: bold;
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
    }
    .conversation-container {
        max-height: 400px;
        overflow-y: auto;
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    .user-message {
        background-color: #E3F2FD;
        padding: 0.75rem;
        border-radius: 15px 15px 0 15px;
        margin-bottom: 0.5rem;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #F5F5F5;
        padding: 0.75rem;
        border-radius: 15px 15px 15px 0;
        margin-bottom: 0.5rem;
        width: fit-content;
        max-width: 80%;
    }
    .controls-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    .logo-container {
        text-align: center;
        margin-bottom: 1rem;
    }
    .logo {
        width: 180px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'is_speaking' not in st.session_state:
    st.session_state.is_speaking = False
if 'language' not in st.session_state:
    st.session_state.language = "en"
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Audio queue for processing
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()

# Header with logo
st.markdown("""
<div class="logo-container">
    <img src="https://s3-eu-west-1.amazonaws.com/tpd/logos/60d3a0bc65022800013b18b3/0x0.png" class="logo">
    <h1 class="main-header">Voice Assistant</h1>
    <p class="sub-header">Your AI-powered multilingual voice companion</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input("Groq API Key", value="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", type="password")
    
    # Language selection
    languages = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Japanese": "ja",
        "Chinese": "zh",
        "Hindi": "hi",
        "Arabic": "ar",
        "Korean": "ko"
    }
    
    language_name = st.selectbox("Select Language", list(languages.keys()))
    st.session_state.language = languages[language_name]
    
    # Microphone sensitivity
    mic_sensitivity = st.slider("Microphone Sensitivity", min_value=300, max_value=6000, value=4000, step=100)
    pause_threshold = st.slider("Pause Threshold", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
    
    # Download conversation
    if st.button("Download Conversation"):
        conversation_md = "# Voice Assistant Conversation\n\n"
        conversation_md += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        conversation_md += f"Language: {language_name}\n\n"
        
        for entry in st.session_state.conversation:
            conversation_md += f"## User\n{entry['user']}\n\n"
            conversation_md += f"## Assistant\n{entry['assistant']}\n\n"
            conversation_md += "---\n\n"
        
        # Create a download link
        b64 = base64.b64encode(conversation_md.encode()).decode()
        href = f'<a href="data:text/markdown;base64,{b64}" download="conversation.md">Click to download conversation as Markdown</a>'
        st.markdown(href, unsafe_allow_html=True)

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    # Conversation display
    st.markdown("<h3>Conversation</h3>", unsafe_allow_html=True)
    
    conversation_container = st.empty()
    
    # Display conversation
    conversation_html = '<div class="conversation-container">'
    for entry in st.session_state.conversation:
        conversation_html += f'<div class="user-message">{entry["user"]}</div>'
        conversation_html += f'<div class="assistant-message">{entry["assistant"]}</div>'
    conversation_html += '</div>'
    
    conversation_container.markdown(conversation_html, unsafe_allow_html=True)

with col2:
    # Controls
    st.markdown('<div class="controls-container">', unsafe_allow_html=True)
    
    status_container = st.empty()
    
    if st.session_state.is_listening:
        status_container.markdown('<p class="status-listening">Listening...</p>', unsafe_allow_html=True)
    elif st.session_state.is_speaking:
        status_container.markdown('<p class="status-processing">Speaking...</p>', unsafe_allow_html=True)
    else:
        status_container.markdown('<p>Ready</p>', unsafe_allow_html=True)
    
    if st.button("Start Listening"):
        st.session_state.is_listening = True
        status_container.markdown('<p class="status-listening">Listening...</p>', unsafe_allow_html=True)
    
    if st.button("Stop Listening"):
        st.session_state.is_listening = False
        status_container.markdown('<p>Ready</p>', unsafe_allow_html=True)
    
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.conversation_history = []
        conversation_container.markdown('<div class="conversation-container"></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 Voice Assistant | Powered by Groq, Whisper, and Streamlit</p>
</div>
""", unsafe_allow_html=True)

# --------------------- VOICE ASSISTANT FUNCTIONS ---------------------

def init_groq_client():
    """Initialize Groq client with API key"""
    return Groq(api_key=api_key)

def transcribe_with_groq(audio_path, language="en"):
    """Transcribe audio using Groq's Whisper API"""
    try:
        client = init_groq_client()
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

def get_llama_response(question, language="en"):
    """Get response from Groq's Llama model with conversation history"""
    try:
        client = init_groq_client()
        # Create system prompt based on selected language
        system_prompt = f"You are a helpful, friendly voice assistant. Keep your responses concise and conversational. Respond in the same language as the user's question."
        
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

def speak_text(text, language="en"):
    """Convert text to speech and play it"""
    try:
        st.session_state.is_speaking = True
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts_filename = tmp_file.name
        
        # Generate speech
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(tts_filename)
        
        # Play the audio
        playsound(tts_filename)
        
        # Clean up the temporary file
        if os.path.exists(tts_filename):
            os.remove(tts_filename)
            
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")
    finally:
        st.session_state.is_speaking = False

def save_audio_data(audio_data):
    """Save audio data to a temporary file and return the filename"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        temp_filename = tmp_file.name
    
    with open(temp_filename, "wb") as f:
        f.write(audio_data.get_wav_data())
    
    return temp_filename

def process_audio_file(audio_file):
    """Process an audio file through the entire pipeline"""
    # Step 1: Transcribe the audio
    transcription = transcribe_with_groq(audio_file, st.session_state.language)
    
    if not transcription:
        st.warning("No speech detected or transcription failed")
        return
    
    # Step 2: Get AI response
    response = get_llama_response(transcription, st.session_state.language)
    
    # Step 3: Add to conversation display
    st.session_state.conversation.append({"user": transcription, "assistant": response})
    
    # Update conversation display
    conversation_html = '<div class="conversation-container">'
    for entry in st.session_state.conversation:
        conversation_html += f'<div class="user-message">{entry["user"]}</div>'
        conversation_html += f'<div class="assistant-message">{entry["assistant"]}</div>'
    conversation_html += '</div>'
    
    conversation_container.markdown(conversation_html, unsafe_allow_html=True)
    
    # Step 4: Speak the response
    speak_text(response, st.session_state.language)

def audio_processor_thread():
    """Background thread to process audio from the queue"""
    while True:
        if not st.session_state.audio_queue.empty():
            audio_data = st.session_state.audio_queue.get()
            if audio_data is None:  # None is our signal to exit
                break
                
            try:
                # Save the audio data to a temporary file
                temp_filename = save_audio_data(audio_data)
                
                # Process the audio file
                process_audio_file(temp_filename)
                
                # Clean up
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
            
            # Mark the task as done
            st.session_state.audio_queue.task_done()
        
        # Sleep to avoid consuming too much CPU
        time.sleep(0.1)

def callback(recognizer, audio):
    """Callback function for when audio is detected"""
    # If we're currently speaking or not listening, don't process this audio
    if st.session_state.is_speaking or not st.session_state.is_listening:
        return
        
    st.session_state.audio_queue.put(audio)

# --------------------- MAIN APP LOGIC ---------------------

def main():
    # Initialize recognizer and microphone
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    # Start the audio processor thread
    if 'processor_thread' not in st.session_state:
        st.session_state.processor_thread = threading.Thread(target=audio_processor_thread)
        st.session_state.processor_thread.daemon = True
        st.session_state.processor_thread.start()
    
    # Set the speech energy threshold for better detection
    recognizer.energy_threshold = mic_sensitivity
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = pause_threshold
    
    # Set up the microphone and start listening if needed
    if st.session_state.is_listening and 'stop_listening' not in st.session_state:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        
        st.session_state.stop_listening = recognizer.listen_in_background(microphone, callback)
    
    # Stop listening if needed
    if not st.session_state.is_listening and 'stop_listening' in st.session_state:
        st.session_state.stop_listening(wait_for_stop=False)
        del st.session_state.stop_listening

if __name__ == "__main__":
    main()
