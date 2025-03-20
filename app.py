import os
import tempfile
import streamlit as st
from gtts import gTTS
from datetime import datetime
from groq import Groq
import audio_recorder_streamlit as ast

# ------------------ PAGE CONFIGURATION ------------------
st.set_page_config(page_title="AI Voice Assistant", page_icon="🎙️", layout="wide")

# ------------------ CUSTOM CSS ------------------
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
if 'language' not in st.session_state:
    st.session_state.language = "en"
if 'audio_data' not in st.session_state:
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

# ------------------ FUNCTIONS ------------------
def init_groq_client(api_key):
    return Groq(api_key=api_key)

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
            
            Always respond in a way that is clear,as the question is necessary, engaging, and easy to understand when spoken aloud.
            """

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}]
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.5,
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
    """Processes a single audio message, gets AI response, and generates audio."""
    if not audio_bytes:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        audio_file = tmp_file.name

    try:
        transcription = transcribe_with_groq(audio_file, api_key, st.session_state.language)
        if not transcription:
            st.warning("No speech detected or transcription failed")
            return None, None

        response = get_llama_response(transcription, api_key, st.session_state.language)
        audio_response = text_to_speech(response, st.session_state.language)

        return transcription, response, audio_response

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None
    finally:
        os.remove(audio_file)

# ------------------ APP LAYOUT ------------------
st.markdown("<h1 class='main-header'>AI Voice Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Record a message, and the AI will respond.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Conversation")
    conversation_container = st.container()
    with conversation_container:
        st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
        for entry in st.session_state.conversation:
            st.markdown(f'<div class="user-message">{entry["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message">{entry["assistant"]}</div>', unsafe_allow_html=True)
            if "audio_response" in entry:
                st.audio(entry["audio_response"], format="audio/mp3")  # Play audio
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("Controls")
    api_key = st.text_input("Groq API Key", type="password")
    language_options = [f"{data['flag']} {name}" for name, data in languages.items()]
    selected_language = st.selectbox("Select Language", language_options, index=0)
    selected_language_name = selected_language.split(" ", 1)[1]
    st.session_state.language = languages[selected_language_name]["code"]

    audio_bytes = ast.audio_recorder(
        text="Click to record",
        recording_color="#e53935",
        neutral_color="#2E5BFF",
        icon_size="2x"
    )

    if audio_bytes and st.session_state.audio_data != audio_bytes:
        st.session_state.audio_data = audio_bytes  # Only update on new recording
        with st.spinner("Processing your message..."):
            transcription, response, audio_response = process_audio(audio_bytes, api_key)
            if transcription and response and audio_response:
                st.session_state.conversation.append({
                    "user": transcription,
                    "assistant": response,
                    "audio_response": audio_response
                })
        st.rerun()  # Rerun after processing

    if st.button("🔄 Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.audio_data = None  # Reset audio data
        st.rerun()
