import os
import tempfile
import streamlit as st
import audio_recorder_streamlit as ast
from gtts import gTTS
from playsound import playsound
from groq import Groq
from deep_translator import GoogleTranslator
import base64
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="AI Voice Assistant",
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
    .recording-status {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        text-align: center;
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

# Header with logo
st.markdown("""
<div class="logo-container">
    <h1 class="main-header">AI Voice Assistant</h1>
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
        "Chinese": "zh-CN",
        "Hindi": "hi",
        "Arabic": "ar",
        "Korean": "ko"
    }
    
    language_name = st.selectbox("Select Language", list(languages.keys()))
    st.session_state.language = languages[language_name]
    
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
    
    conversation_container = st.container()
    
    with conversation_container:
        # Display conversation
        for entry in st.session_state.conversation:
            st.markdown(f'<div class="user-message">{entry["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message">{entry["assistant"]}</div>', unsafe_allow_html=True)

with col2:
    # Controls
    st.markdown('<div class="controls-container">', unsafe_allow_html=True)
    
    status_container = st.empty()
    
    if st.session_state.recording_state == 'recording':
        status_container.markdown('<p class="status-listening">Recording...</p>', unsafe_allow_html=True)
    elif st.session_state.is_processing:
        status_container.markdown('<p class="status-processing">Processing...</p>', unsafe_allow_html=True)
    else:
        status_container.markdown('<p>Ready</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎙️ Start Recording", 
                   type="primary" if st.session_state.recording_state != 'recording' else "secondary",
                   disabled=st.session_state.recording_state == 'recording'):
            st.session_state.recording_state = 'recording'
            st.session_state.audio_bytes = None
            st.experimental_rerun()
    
    with col2:
        if st.button("⏹️ Stop Recording", 
                   type="primary" if st.session_state.recording_state == 'recording' else "secondary",
                   disabled=st.session_state.recording_state != 'recording'):
            st.session_state.recording_state = 'stopped'
            st.experimental_rerun()
    
    if st.button("🔄 Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.conversation_history = []
        st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

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

def translate_text(text, target_language="en"):
    """Translate text using Google Translator"""
    try:
        translator = GoogleTranslator(source='auto', target=target_language)
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        st.error(f"Error in translation: {str(e)}")
        return text

def save_audio_to_file(audio_bytes):
    """Save audio bytes to a temporary file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        return tmp_file.name

def text_to_speech(text, language="en"):
    """Convert text to speech and return audio file path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts_filename = tmp_file.name
        
        # Generate speech
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(tts_filename)
        
        return tts_filename
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")
        return None

def process_audio(audio_bytes):
    """Process audio bytes through the voice assistant pipeline"""
    st.session_state.is_processing = True
    
    try:
        # Save audio to file
        audio_file = save_audio_to_file(audio_bytes)
        
        # Transcribe audio
        transcription = transcribe_with_groq(audio_file, st.session_state.language)
        
        if not transcription:
            st.warning("No speech detected or transcription failed")
            return
        
        # Get AI response
        response = get_llama_response(transcription, st.session_state.language)
        
        # Add to conversation display
        st.session_state.conversation.append({"user": transcription, "assistant": response})
        
        # Generate audio response
        audio_response = text_to_speech(response, st.session_state.language)
        
        # Clean up audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        return audio_response
    
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None
    
    finally:
        st.session_state.is_processing = False

# --------------------- AUDIO RECORDING SECTION ---------------------

if st.session_state.recording_state == 'recording':
    st.markdown("""<div class="recording-status" style="background-color: #ff4b4b; color: white;"> Recording in progress... 🎙️ </div>""", unsafe_allow_html=True)
    
    # Use audio_recorder_streamlit to record audio
    audio_bytes = ast.audio_recorder(pause_threshold=2.0, sample_rate=44100)
    
    if audio_bytes:
        st.session_state.audio_bytes = audio_bytes
        st.session_state.recording_state = 'stopped'
        st.experimental_rerun()

# Process recorded audio
if st.session_state.audio_bytes and not st.session_state.is_processing:
    # Display the recorded audio
    st.audio(st.session_state.audio_bytes, format="audio/wav")
    
    # Process the audio
    with st.spinner("Processing your message..."):
        audio_response = process_audio(st.session_state.audio_bytes)
        
        if audio_response:
            # Play the audio response
            st.audio(audio_response, format="audio/mp3")
            
            # Clean up the audio response file
            if os.path.exists(audio_response):
                os.remove(audio_response)
    
    # Clear the audio bytes to prevent reprocessing
    st.session_state.audio_bytes = None

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 AI Voice Assistant | Powered by Groq, Whisper, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
