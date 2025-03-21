import os
import tempfile
import streamlit as st
import base64
from gtts import gTTS
from datetime import datetime
from groq import Groq
import audio_recorder_streamlit as ast

# ------------------ PAGE CONFIGURATION ------------------
st.set_page_config(page_title="AI Voice Assistant", page_icon="🎙️", layout="wide")


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

     /* Primary and secondary button styles */
    .stButton[data-baseweb="button"] > button[kind="primary"] {
        background-color: #2E5BFF; /* Your primary color */
        color: white;
    }
    .stButton[data-baseweb="button"] > button[kind="secondary"] {
        background-color: #f0f2f6; /* Light gray or your secondary color */
        color: #2E5BFF;
        border: 1px solid #2E5BFF;
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
    
    .audio-buttons {
        display: flex;
        align-items: center;
    }
    .audio-buttons > * {
        margin-right: 0.5rem; /* Add some spacing between buttons */
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
if 'recording_state' not in st.session_state:  # Use a string to represent state
    st.session_state.recording_state = 'idle'  # 'idle', 'recording', 'stopped'
if 'audio_to_autoplay' not in st.session_state:
    st.session_state.audio_to_autoplay = None
if 'language_error' not in st.session_state:
    st.session_state.language_error = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = ""

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
                        You are an engaging and expressive AI voice assistant, designed to provide natural and fluid spoken responses in {language_name}.
                        Your responses should be:
                        - **Concise and short**: Keep answers brief but meaningful.
                        - **Conversational**: Sound natural, as if speaking to a human.
                        - **Insightful**: Offer thoughtful and relevant responses.
                        - **Expressive**: Adapt tone to match the context of the question.
                        You will be asked general and reflective questions: Answer such questions politefully
                        Always respond in a way that is clear, as the question is necessary, engaging, and easy to understand when spoken aloud. In short and specific on to point !
                        """

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}]
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.5,
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
st.logo("https://s3-eu-west-1.amazonaws.com/tpd/logos/60d3a0bc65022800013b18b3/0x0.png")
# Header with Logo
st.markdown("<div class='logo-container'><img src='https://s3-eu-west-1.amazonaws.com/tpd/logos/60d3a0bc65022800013b18b3/0x0.png'><h1>AI Voice Assistant</h1></div>", unsafe_allow_html=True)
st.sidebar.title("App Instructions")
st.sidebar.markdown(
    """
    **Welcome to the AI Voice Assistant!**

    **How It Works:**
    - **Record Your Message:** Click **Start Recording** and speak into your microphone.
    - **Stop Recording:** Press **Stop** when you’re finished.
    - **Processing:** Your voice will be transcribed using Groq's Whisper API, and an AI response will be generated.
    - **Language Selection:** Choose your preferred language from the dropdown.
    - **Clear Conversation:** Use the **Clear Conversation** button to reset the chat history.

    **Setup:**
    - Enter your **Groq API Key** in the provided field.
    - Ensure you have a stable internet connection for transcription and AI responses.
    """
)

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Conversation")
    with st.container():
        st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
        for entry in st.session_state.conversation:
            st.markdown(f'<div class="user-message">{entry["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message">{entry["assistant"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("Controls")
    api_key = st.text_input("Groq API Key", type="password",value = "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")
    st.link_button(label="Get API Here", url="https://console.groq.com/playground")
    language_options = [f"{data['flag']} {name}" for name, data in languages.items()]
    selected_language = st.selectbox("Select Language", language_options, index=0)
    selected_language_name = selected_language.split(" ", 1)[1]
    st.session_state.language = languages[selected_language_name]["code"]

    # Use columns for Start/Stop buttons
    col_rec, col_stop = st.columns([1, 1])

    with col_rec:
        if st.button("🎙️ Start Recording",
                     type="primary" if st.session_state.recording_state != 'recording' else "secondary",
                     disabled=st.session_state.recording_state == 'recording'):
            st.session_state.recording_state = 'recording'
            st.session_state.audio_data = None  # Clear previous data
            st.session_state.language_error = False  # Reset error state
            st.session_state.error_message = "" # Clear error message
            st.rerun()

    with col_stop:
        if st.button("⏹️ Stop",
                     type="primary" if st.session_state.recording_state == 'recording' else "secondary",
                     disabled=st.session_state.recording_state != 'recording'):
            st.session_state.recording_state = 'stopped'
            st.rerun()

    if st.session_state.recording_state == 'recording':
        audio_bytes = ast.audio_recorder(
            text="Click -> Speak -> Press 'Stop'",
            recording_color="#e53935",
            neutral_color="#2E5BFF",
            icon_size="2x",
            key="audio_recorder"
        )
        if audio_bytes and st.session_state.audio_data != audio_bytes:
            st.session_state.audio_data = audio_bytes

    if st.session_state.audio_data and st.session_state.recording_state == 'stopped':
        with st.spinner("Processing your message..."):
            transcription, response, audio_response = process_audio(st.session_state.audio_data, api_key)
            if transcription and response and audio_response:
                st.session_state.conversation.append({
                    "user": transcription,
                    "assistant": response,
                    "audio_response": audio_response
                })
                st.session_state.audio_to_autoplay = audio_response
            st.session_state.audio_data = None  # Clear audio data after processing
            st.session_state.recording_state = 'idle'  # Reset to idle after processing
        st.rerun()

    if st.button("🔄 Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.audio_data = None
        st.session_state.audio_to_autoplay = None
        st.session_state.recording_state = 'idle'  # Reset recording state
        st.rerun()

# Auto-play audio if available
if st.session_state.audio_to_autoplay:
    autoplay_audio(st.session_state.audio_to_autoplay)
    st.session_state.audio_to_autoplay = None

# Footer
st.markdown("<div class='footer'>© 2024 Home.LLC | <a href='https://www.home.llc/'>Visit our website</a></div>", unsafe_allow_html=True)
