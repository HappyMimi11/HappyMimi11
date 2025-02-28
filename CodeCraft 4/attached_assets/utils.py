import io
from typing import Dict, List, Tuple, Optional
import streamlit as st
from faster_whisper import WhisperModel
from streamlit.runtime.uploaded_file_manager import UploadedFile  # Fix UploadedFile import

SUPPORTED_FORMATS = [
    "mp3", "wav", "aac", "flac", "ogg", "m4a", 
    "mp4", "wma", "amr", "webm"
]

MODELS = {
    "tiny.en": {"accuracy": "85%-94%", "speed": "±10x speed"},
    "base.en": {"accuracy": "90%-96%", "speed": "±7x speed"},
    "small.en": {"accuracy": "93%-97%", "speed": "±4x speed"}
}

@st.cache_resource
def load_model(model_name: str) -> WhisperModel:
    """Load and cache the Whisper model."""
    return WhisperModel(model_name, device="cpu", compute_type="int8")

def validate_files(files: List[UploadedFile]) -> List[Tuple[bool, str, UploadedFile]]:
    """Validate multiple uploaded files."""
    results = []
    for file in files:
        # Check file size (200MB limit)
        if file.size > 200 * 1024 * 1024:
            results.append((False, f"{file.name}: File size exceeds 200MB limit", file))
            continue

        # Check file extension
        file_ext = file.name.split('.')[-1].lower()
        if file_ext not in SUPPORTED_FORMATS:
            results.append((False, f"{file.name}: Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}", file))
            continue

        results.append((True, f"{file.name}: Valid file", file))
    return results

def transcribe_audio_bytes(audio_bytes: bytes, model: WhisperModel, progress_bar) -> Tuple[bool, str]:
    """Transcribe audio directly from bytes."""
    try:
        # Create a single file-like object in memory
        audio_data = io.BytesIO(audio_bytes)

        # Update progress to show processing has started
        progress_bar.progress(0.3)

        # Transcribe the entire audio file
        segments, info = model.transcribe(
            audio_data,
            beam_size=5,
            word_timestamps=False
        )

        # Check if any segments were generated
        if not segments:
            return False, "No speech detected in the audio file"

        # Combine all segments
        transcript = " ".join([segment.text.strip() for segment in segments])

        # Update progress to show completion
        progress_bar.progress(1.0)

        if not transcript:
            return False, "Transcription resulted in empty text"

        return True, transcript

    except RuntimeError as e:
        return False, f"Runtime error during transcription: {str(e)}"
    except ValueError as e:
        return False, f"Invalid audio format: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error during transcription: {str(e)}"

def save_transcription(text: str, filename: str) -> Tuple[bool, str]:
    """Save transcription to a text file."""
    try:
        output_filename = f"transcription_{filename.rsplit('.', 1)[0]}.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text)
        return True, output_filename
    except Exception as e:
        return False, f"Error saving transcription: {str(e)}"