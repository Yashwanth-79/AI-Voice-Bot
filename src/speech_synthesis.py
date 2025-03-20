import tempfile
import os
from gtts import gTTS

def text_to_speech(text, language="en"):
    """Converts text to speech using gTTS and returns audio bytes."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(tmp_file.name)
            with open(tmp_file.name, "rb") as f:
                audio_bytes = f.read()
        os.remove(tmp_file.name)
        return audio_bytes
    except Exception as e:
        print(f"Error in text-to-speech: {e}") # Log error
        return None
      
