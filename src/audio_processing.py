import tempfile
import base64
from .language_model import get_llama_response
from .speech_synthesis import text_to_speech
from groq import Groq

def transcribe_with_groq(audio_path, api_key, language="en"):
    """Transcribes audio using Groq's Whisper API."""
    try:
        client = Groq(api_key=api_key)
        with open(audio_path, "rb") as file:
            response = client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3-turbo",
                language=language
            )
        return response.text.strip()
    except Exception as e:
        print(f"Error in transcription: {e}")  # Log the error
        return ""

def process_audio(audio_bytes, api_key, language="en"):
    """Processes audio, gets AI response, generates audio response."""
    if not audio_bytes:
        return None, None, None

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        audio_file = tmp_file.name

    try:
        transcription = transcribe_with_groq(audio_file, api_key, language)
        if not transcription:
            print("No speech detected or transcription failed") # Log warning
            return None, None, None

        response = get_llama_response(transcription, api_key, language)
        audio_response = text_to_speech(response, language)

        return transcription, response, audio_response

    except Exception as e:
        print(f"Error processing audio: {e}") # Log error
        return None, None, None
    finally:
        os.remove(audio_file)

def generate_autoplay_html(audio_bytes):
    """Generates HTML for autoplaying audio."""
    if not audio_bytes:
        return ""
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio controls autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    return md
