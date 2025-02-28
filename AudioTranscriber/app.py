import os
import streamlit as st
from utils import (
    MODELS, validate_files, transcribe_audio_bytes,
    save_transcription, load_model
)

SUPPORTED_FORMATS = ["mp3", "wav", "aac", "flac", "ogg", "m4a", "mp4", "wma", "amr", "webm"]

def main():
    try:
        # Set page config
        st.set_page_config(
            page_title="Audio Transcription App",
            page_icon="🎤",
            layout="wide"
        )

        # Load custom CSS
        if os.path.exists("style.css"):
            with open("style.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        st.title("🎤 Audio Transcription App")
        st.markdown("Convert your audio files to text using Whisper AI! 🤖")

        # Model selection
        st.subheader("1. Select Transcription Model")
        col1, col2 = st.columns([1, 2])

        with col1:
            model_name = st.selectbox(
                "Model",  # Added label to fix accessibility warning
                options=list(MODELS.keys()),
                format_func=lambda x: x.split('.')[0].capitalize()
            )

        with col2:
            st.markdown(
                f"""
                <div class="model-info">
                <strong>{model_name}</strong><br>
                Accuracy: {MODELS[model_name]['accuracy']}<br>
                Speed: {MODELS[model_name]['speed']}
                </div>
                """,
                unsafe_allow_html=True
            )

        # File upload
        st.subheader("2. Upload Audio Files")
        uploaded_files = st.file_uploader(
            "Audio Files",  # Added label to fix accessibility warning
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            help="Drop your audio files here or click to upload"
        )

        if uploaded_files:
            # Validate files
            validation_results = validate_files(uploaded_files)
            valid_files = [(f, msg) for valid, msg, f in validation_results if valid]
            invalid_files = [(f, msg) for valid, msg, f in validation_results if not valid]

            # Show invalid files if any
            if invalid_files:
                st.error("Invalid files detected:")
                for _, msg in invalid_files:
                    st.warning(msg)

            # Process valid files
            if valid_files and st.button("Start Transcription", help="Click to begin transcription"):
                try:
                    # Load model once for all files
                    with st.spinner("Loading model..."):
                        model = load_model(model_name)

                    # Create columns for parallel display
                    cols = st.columns(min(len(valid_files), 2))

                    # Process files
                    for idx, (file, _) in enumerate(valid_files):
                        col = cols[idx % 2]
                        with col:
                            st.markdown(f"### {file.name}")
                            progress_bar = st.progress(0)

                            success, result = transcribe_audio_bytes(
                                file.getvalue(),
                                model,
                                progress_bar
                            )

                            if success:
                                st.markdown(f"#### Transcription for {file.name}:")
                                st.text_area(
                                    f"Transcription result for {file.name}",  # Added label to fix accessibility warning
                                    result,
                                    height=150,
                                    key=f"result_{file.name}"
                                )

                                # Save transcription
                                save_success, save_path = save_transcription(result, file.name)
                                if save_success:
                                    with open(save_path, 'r') as f:
                                        st.download_button(
                                            f"Download {file.name} Transcription",
                                            f.read(),
                                            file_name=save_path,
                                            mime="text/plain",
                                            key=f"download_{file.name}"
                                        )
                                else:
                                    st.error(save_path)
                            else:
                                st.error(f"Error processing {file.name}: {result}")
                except Exception as e:
                    st.error(f"Error during transcription: {str(e)}")

        # Display supported formats
        st.markdown("---")
        st.markdown("### Tips")
        st.markdown("1. Optimised for English-only.")
        st.markdown("2. Try to keep the background noise to a minimum.")
        st.markdown("3. Transcription accuracy is mostly based on your accent and clearness of articulation.")
                    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please try refreshing the page. If the error persists, contact support.")

if __name__ == "__main__":
    main()
