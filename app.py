import os
import tempfile
import streamlit as st
import base64
from gtts import gTTS
from datetime import datetime
from groq import Groq
import audio_recorder_streamlit as ast

# ------------------ PAGE CONFIGURATION ------------------
st.set_page_config(page_title="AI Voice Assistant", page_icon="🎙️")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main container styling */
    .stApp {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }

    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E5BFF;
        margin-bottom: 0;
        text-align: center;
        padding: 1rem 0;
        background-color: #f0f2f6; /* Light background for header */
        border-bottom: 2px solid #2E5BFF; /* Blue border */
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #4F4F4F;
        margin-top: 0;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 300;
    }

    /* Content area */
    .content-area {
        flex-grow: 1; /* Takes up remaining space */
        display: flex; /* Use flexbox for layout */
        flex-direction: row; /* Columns */
        padding: 2rem;
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
    /* footer {visibility: hidden;}  Hide default footer */

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

    /* Custom Footer */
    .footer {
        background-color: #2E5BFF;
        color: white;
        text-align: center;
        padding: 1rem 0;
        width: 100%;
    }
    .footer a {
        color: white;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .logo-container {
        display: flex;
        justify-content: center; /* Center horizontally */
        align-items: center;     /* Center vertically */
        padding: 1rem;
    }
    .logo-container img {
        max-height: 50px; /* Adjust as needed */
        margin-right: 1rem;
    }
    
    /* Recording controls */
    .recording-controls {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .stop-button {
        background-color: #e53935;
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .stop-button:hover {
        background-color: #c62828;
        transform: scale(1.05);
    }
    
    .stop-icon {
        width: 16px;
        height: 16px;
        background-color: white;
    }

</style>
""", unsafe_allow_html=True)

# ------------------ INITIALIZE SESSION STATE ------------------
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'language' not in st.session_state:
    st.session_state.language = "en"
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'audio_to_autoplay' not in st.session_state:
    st.session_state.audio_to_autoplay = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

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
    return Groq(api_key=api_key)

def autoplay_audio(audio_bytes):
    """Function to auto-play audio using HTML audio tag with autoplay attribute"""
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

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
        system_prompt = f"""
            You are a highly engaging and expressive AI voice assistant, designed to provide natural and fluid spoken responses in {language_name}.
            Your responses should be:
            - **Concise**: Keep answers brief but meaningful.
            - **Conversational**: Sound natural, as if speaking to a human.
            - **Insightful**: Offer thoughtful and relevant responses.
            - **Expressive**: Adapt tone to match the context of the question.
            You will be asked general and reflective questions: Answer such questions politefully
            Always respond in a way that is clear, as the question is necessary, engaging, and easy to understand when spoken aloud.
            """

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}]
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.5,  # Adjusted for more focused responses
            max_tokens=150,
            top_p=1,
            stream=False
        )
        return completion.choices[0].message.content.strip()
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

def process_audio(audio_bytes, api_key):
    """Processes audio, gets AI response, generates audio response."""
    if not audio_bytes:
        return None, None, None

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        audio_file = tmp_file.name

    try:
        transcription = transcribe_with_groq(audio_file, api_key, st.session_state.language)
        if not transcription:
            st.warning("No speech detected or transcription failed")
            return None, None, None

        response = get_llama_response(transcription, api_key, st.session_state.language)
        audio_response = text_to_speech(response, st.session_state.language)

        return transcription, response, audio_response

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None
    finally:
        os.remove(audio_file)

# ------------------ APP LAYOUT ------------------

# Header with Logo
st.markdown("<div class='logo-container'><img src='https://s3-eu-west-1.amazonaws.com/tpd/logos/60d3a0bc65022800013b18b3/0x0.png'><h1>AI Voice Assistant</h1></div>", unsafe_allow_html=True)

# Main content area (using columns for layout)
col1, col2 = st.columns([3, 1])  # Adjust column widths as needed

with col1:
    st.subheader("Conversation")
    conversation_container = st.container()
    with conversation_container:
        st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
        for i, entry in enumerate(st.session_state.conversation):
            st.markdown(f'<div class="user-message">{entry["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message">{entry["assistant"]}</div>', unsafe_allow_html=True)
            
            # Create a unique key for each audio entry
            audio_key = f"audio_{i}"
            if "audio_response" in entry and entry["audio_response"]:
                # Each conversation entry gets its own autoplay audio
                st.markdown(f"<div id='{audio_key}'>", unsafe_allow_html=True)
                b64 = base64.b64encode(entry["audio_response"]).decode()
                md = f"""
                    <audio controls autoplay="true">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                    """
                st.markdown(md, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("Controls")
    api_key = st.text_input("Groq API Key", type="password")
    st.link_button(label="Get API Here", url="https://console.groq.com/playground")
    language_options = [f"{data['flag']} {name}" for name, data in languages.items()]
    selected_language = st.selectbox("Select Language", language_options, index=0)
    selected_language_name = selected_language.split(" ", 1)[1]
    st.session_state.language = languages[selected_language_name]["code"]

    # Recording controls with Stop button
    st.markdown("<div class='recording-controls'>", unsafe_allow_html=True)
    
    # Start recording button
    col_rec1, col_rec2 = st.columns([3, 1])
    with col_rec1:
        audio_bytes = ast.audio_recorder(
            text="Click to record",
            recording_color="#e53935",
            neutral_color="#2E5BFF",
            icon_size="2x",
            pause_threshold=120.0,  # Long pause threshold to allow stopping manually
            recording_callback=lambda: setattr(st.session_state, 'is_recording', True),
            stopped_callback=lambda: setattr(st.session_state, 'is_recording', False)
        )
    
    with col_rec2:
        # Stop button
        if st.button("⏹️ Stop", key="stop_recording"):
            st.session_state.is_recording = False
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

    if audio_bytes and st.session_state.audio_data != audio_bytes:
        st.session_state.audio_data = audio_bytes
        with st.spinner("Processing your message..."):
            transcription, response, audio_response = process_audio(audio_bytes, api_key)
            if transcription and response and audio_response:
                # Add to conversation history with audio response
                st.session_state.conversation.append({
                    "user": transcription,
                    "assistant": response,
                    "audio_response": audio_response
                })
                # Set the audio to autoplay only for the most recent conversation
                st.session_state.audio_to_autoplay = audio_response
        st.rerun()

    if st.button("🔄 Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.audio_data = None
        st.session_state.audio_to_autoplay = None
        st.rerun()

# Auto-play most recent audio if available 
# (This is a backup and may not be necessary with the in-conversation autoplays)
if st.session_state.audio_to_autoplay:
    autoplay_audio(st.session_state.audio_to_autoplay)
    st.session_state.audio_to_autoplay = None  # Reset after playing

# Footer
st.markdown("<div class='footer'>© 2024 Home.LLC | <a href='https://www.home.llc/'>Visit our website</a></div>", unsafe_allow_html=True)
