import os
import time
import tempfile
import streamlit as st
import audio_recorder_streamlit as ast
from gtts import gTTS
from datetime import datetime
from groq import Groq
from playsound import playsound

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
</style>
""", unsafe_allow_html=True)

# ------------------ INITIALIZE SESSION STATE ------------------
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'language' not in st.session_state:
    st.session_state.language = "en"

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

def process_audio(audio_bytes, api_key):
    """Process recorded audio through the pipeline"""
    st.session_state.is_processing = True
    
    try:
        # Save audio to file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            audio_file = tmp_file.name
        
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

# ------------------ APP LAYOUT ------------------
# Header
st.markdown("<h1 class='main-header'>AI Voice Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Your AI-powered multilingual voice companion</p>", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([3, 1])

with col1:
    # Conversation display
    st.subheader("Conversation")
    
    # Display the conversation history
    st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
    for entry in st.session_state.conversation:
        time_str = entry.get("timestamp", "")
        st.markdown(f'<div class="user-message">{entry["user"]}<div class="message-timestamp">{time_str}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="assistant-message">{entry["assistant"]}<div class="message-timestamp">{time_str}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Settings and controls
    st.subheader("Controls")
    
    # API Key input
    api_key = st.text_input("Groq API Key", value="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", type="password")
    
    # Language selection
    language_options = [f"{data['flag']} {name}" for name, data in languages.items()]
    selected_language = st.selectbox("Select Language", language_options, index=0)
    
    # Extract language code from selection
    selected_language_name = selected_language.split(" ", 1)[1]
    st.session_state.language = languages[selected_language_name]["code"]
    
    # Status indicator
    if st.session_state.is_processing:
        st.info("Processing... ⏳")
    else:
        st.success("Ready 🎤")
    
    # Audio recorder using audio_recorder_streamlit
    st.write("Click to record")
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
        st.rerun()

# Process recorded audio
if audio_bytes and not st.session_state.is_processing:
    # Display the recorded audio
    st.sidebar.audio(audio_bytes, format="audio/wav")
    
    # Process the audio
    with st.spinner("Processing your message..."):
        transcription, audio_response = process_audio(audio_bytes, api_key)
        
        if transcription and audio_response:
            # Play the audio response
            st.sidebar.audio(audio_response, format="audio/mp3")
            st.rerun()
