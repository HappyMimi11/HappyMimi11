import io
import os
import tempfile
from typing import Dict, List, Tuple, Optional
import streamlit as st
from faster_whisper import WhisperModel
from streamlit.runtime.uploaded_file_manager import UploadedFile
import subprocess
brew install ffmpeg 

SUPPORTED_FORMATS = [
    "mp3", "wav", "aac", "flac", "ogg", "m4a", 
    "mp4", "wma", "amr", "webm"
]

MODELS = {
    "large-v3-turbo": {"accuracy": "94%-98%", "speed": "±8x speed"},
    "tiny.en": {"accuracy": "85%-94%", "speed": "±10x speed"},
    "base.en": {"accuracy": "90%-96%", "speed": "±7x speed"},
    "small.en": {"accuracy": "93%-97%", "speed": "±4x speed"},
    "large-v3": {"accuracy": "95%-98%", "speed": "±1x speed"},
    }

@st.cache_resource
def load_model(model_name: str) -> WhisperModel:
    """Load and cache the Whisper model."""
    try:
        return WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        raise

def validate_files(files: List[UploadedFile]) -> List[Tuple[bool, str, UploadedFile]]:
    """Validate multiple uploaded files."""
    results = []
    for file in files:
        # Check file extension
        file_ext = file.name.split('.')[-1].lower()
        if file_ext not in SUPPORTED_FORMATS:
            results.append((False, f"{file.name}: Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}", file))
            continue

        results.append((True, f"{file.name}: Valid file", file))
    return results

def convert_audio(input_bytes: bytes, input_format: str) -> Tuple[bool, bytes]:
    """Convert audio to WAV format using ffmpeg."""
    temp_in = None
    output_path = None
    try:
        temp_in = tempfile.NamedTemporaryFile(suffix=f'.{input_format}', delete=False)
        temp_in.write(input_bytes)
        temp_in.flush()

        output_path = tempfile.mktemp(suffix='.wav')

        command = [
            'ffmpeg',
            '-i', temp_in.name,
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '16000',
            '-f', 'wav',
            output_path,
            '-y'
        ]

        process = subprocess.run(command, capture_output=True, text=True)

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {process.stderr}")

        with open(output_path, 'rb') as f:
            wav_bytes = f.read()

        return True, wav_bytes

    except Exception as e:
        return False, str(e).encode()

    finally:
        # Clean up temporary files
        if temp_in and os.path.exists(temp_in.name):
            os.unlink(temp_in.name)
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)

def transcribe_audio_bytes(audio_bytes, model, progress_bar=None):
    """Transcribe audio directly from bytes using chunked processing."""
    try:
        if progress_bar:
            progress_bar.progress(0.1)

        # Create temporary directory for chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_path = os.path.join(temp_dir, 'input.tmp')
            with open(input_path, 'wb') as f:
                f.write(audio_bytes)

            # Get duration
            probe_cmd = [
                'ffprobe',
                '-i', input_path,
                '-show_entries', 'format=duration',
                '-v', 'quiet',
                '-of', 'csv=p=0'
            ]
            duration_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if duration_result.returncode != 0:
                return False, "Failed to get audio duration"

            total_duration = float(duration_result.stdout.strip())
            chunk_duration = 2400  # 40 minutes in seconds
            num_chunks = int(total_duration / chunk_duration) + 1

            all_segments = []

            for i in range(num_chunks):
                if progress_bar:
                    progress = 0.1 + (0.8 * i / num_chunks)
                    progress_bar.progress(progress)

                chunk_path = os.path.join(temp_dir, f'chunk_{i}.wav')

                # Extract chunk
                start_time = i * chunk_duration
                chunk_cmd = [
                    'ffmpeg',
                    '-i', input_path,
                    '-ss', str(start_time),
                    '-t', str(chunk_duration),
                    '-ac', '1',  # Mono audio
                    '-ar', '16000',  # 16kHz sample rate
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-y',
                    chunk_path
                ]

                chunk_result = subprocess.run(chunk_cmd, capture_output=True, text=True)
                if chunk_result.returncode != 0:
                    continue  # Skip failed chunks

                # Transcribe chunk
                try:
                    segments, info = model.transcribe(
                        chunk_path,
                        beam_size=5,
                        word_timestamps=False,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    all_segments.extend(segments)
                except Exception as e:
                    print(f"Error transcribing chunk {i}: {str(e)}")
                    continue

                # Clean up chunk
                os.remove(chunk_path)

            if not all_segments:
                return False, "No segments were successfully transcribed"

        # Combine all segments
        transcript = " ".join([segment.text.strip() for segment in all_segments])

        if progress_bar:
            progress_bar.progress(1.0)

        if not transcript:
            return False, "Transcription resulted in empty text"

        return True, transcript

    except Exception as e:
        return False, f"Transcription error: {str(e)}"

def save_transcription(text: str, filename: str) -> Tuple[bool, str]:
    """Save transcription to a text file."""
    try:
        output_filename = f"transcription_{filename.rsplit('.', 1)[0]}.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text)
        return True, output_filename
    except Exception as e:
        return False, f"Error saving transcription: {str(e)}"
