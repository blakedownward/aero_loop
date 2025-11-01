import os
import json
import wave
import pandas as pd
import sys
import importlib
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import librosa.display


# set constants
# Resolve repo root based on this file location
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
# Define the path to the raw data folder of audio files to label
RAW_PATH = os.path.join(REPO_ROOT, 'data', 'raw')


# html and css styling
hide_menu_style = """
        <style>
        .block-container {padding-top: 50px;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


def load_show_wave(audio_filepath, fig_width=14, fig_height=5):
    signal, sample_rate = librosa.load(audio_filepath)
    plt.figure(figsize=(fig_width, fig_height))

    return librosa.display.waveshow(signal, sr=sample_rate)


def fetch_unprocessed_batches():
    # list only directories inside RAW_PATH that are not marked processed
    if not os.path.isdir(RAW_PATH):
        return []

    subdirs = [d for d in os.listdir(RAW_PATH) if os.path.isdir(os.path.join(RAW_PATH, d))]
    unprocessed = []
    for batch in subdirs:
        batch_dir = os.path.join(RAW_PATH, batch)
        if ".processed" in os.listdir(batch_dir):
            continue
        unprocessed.append(batch)

    return unprocessed


def fetch_unprocessed_batch(batch: str):
    batch_path = os.path.join(RAW_PATH, batch)
    files = os.listdir(batch_path)

    return [file for file in files if file.endswith(".wav")]


def _load_annotations(batch: str):
    batch_path = os.path.join(RAW_PATH, batch)
    annotation_path = os.path.join(batch_path, 'annotations.json')
    if not os.path.isfile(annotation_path):
        return []
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except json.JSONDecodeError:
        return []


def _save_annotations(batch: str, records: list):
    batch_path = os.path.join(RAW_PATH, batch)
    annotation_path = os.path.join(batch_path, 'annotations.json')
    with open(annotation_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def append_annotation_file(batch: str, data):
    # append or update entry for filename in annotations.json
    records = _load_annotations(batch)
    filename = data.get("filename")
    updated = False
    for i, rec in enumerate(records):
        if rec.get("filename") == filename:
            records[i] = data
            updated = True
            break
    if not updated:
        records.append(data)
    _save_annotations(batch, records)


def _wav_duration_seconds(filepath: str) -> int:
    """Return duration (in whole seconds) for a WAV file.
    Falls back to 1 if duration cannot be determined.
    """
    try:
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate and frames:
                seconds = int(round(frames / float(rate)))
                return max(1, seconds)
    except Exception:
        pass
    return 60

def check_if_annotated(batch: str, filename: str):
    # return True if filename already present in annotations
    batch_path = os.path.join(RAW_PATH, batch)
    annotation_path = os.path.join(batch_path, 'annotations.json')
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return any(d.get("filename") == filename for d in data)
            return False
    except FileNotFoundError:
        return False

unprocessed_batches = fetch_unprocessed_batches()


def _render_all_done():
    st.success("All batches annotated. Ready to process.")
    if st.button("Process Data"):
        with st.spinner("Processing annotated batches..."):
            ok, msg = _run_processor()
        if ok:
            st.success("Processing complete. Outputs in data/processed.")
        else:
            st.error(f"Processing failed: {msg}")


def _run_processor():
    try:
        mlops_path = os.path.join(REPO_ROOT, 'services', 'mlops')
        if mlops_path not in sys.path:
            sys.path.insert(0, mlops_path)
        import processor  # type: ignore
        processor.process()
        return True, "ok"
    except Exception as e:
        return False, str(e)


# Handle case of no batches
if len(unprocessed_batches) == 0:
    _render_all_done()
else:
    # session state for navigation
    if "batch_idx" not in st.session_state:
        st.session_state.batch_idx = 0
    if "file_idx" not in st.session_state:
        st.session_state.file_idx = 0

    # bounds check for batch index
    if st.session_state.batch_idx >= len(unprocessed_batches):
        st.session_state.batch_idx = 0

    target_batch = unprocessed_batches[st.session_state.batch_idx]
    target_batch_path = os.path.join(RAW_PATH, target_batch)
    target_batch_files = fetch_unprocessed_batch(target_batch)

    # filter out already annotated files
    remaining_files = [f for f in target_batch_files if not check_if_annotated(batch=target_batch, filename=f)]

    # if no remaining files -> mark processed and move to next batch
    if len(remaining_files) == 0:
        # mark processed
        open(os.path.join(target_batch_path, '.processed'), 'a').close()
        st.session_state.batch_idx += 1
        st.session_state.file_idx = 0
        # refresh list of batches
        unprocessed_batches = fetch_unprocessed_batches()
        if len(unprocessed_batches) == 0:
            _render_all_done()
        else:
            st.rerun()
    else:
        # clamp file index
        if st.session_state.file_idx >= len(remaining_files):
            st.session_state.file_idx = 0

        file = remaining_files[st.session_state.file_idx]
        filepath = os.path.join(target_batch_path, file)

        head1, head2 = st.columns(2)

        with head1:
            st.subheader(f"Batch: {target_batch}")
            st.header(file)

            with open(filepath, 'rb') as audio_file:
                audio_bytes = audio_file.read()

            clip_len = _wav_duration_seconds(filepath)
            audible_range = st.slider('Ideal audio clip range', 0, clip_len, (0, clip_len), step=1)

            st.audio(audio_bytes, format='audio/wav')

            
            # Default aircraft audible based on clip length:
            # 20s -> False, 60s -> True, others -> True (default)
            classification_default = True
            if clip_len == 20:
                classification_default = False
            elif clip_len == 60:
                classification_default = True
            classification = st.checkbox('Aircraft audible', value=classification_default)
            flag = st.checkbox('Flag clip', value=False)

        with head2:
            x, sr = librosa.load(filepath, sr=None)
            fig, ax = plt.subplots()
            librosa.display.waveshow(x, sr=sr, ax=ax)
            ax.set_yticks([])
            st.pyplot(fig)

            
            
            audible_start = audible_range[0]
            audible_end = audible_range[1]

            data = {
                "filename": file,
                "label": classification,
                "trim_start_s": audible_start,
                "trim_end_s": audible_end,
                "flag": flag
            }

            c1, c2 = st.columns(2)

            with c1:
                # Submit button to commit the labelling data
                if st.button('Commit'):
                    append_annotation_file(batch=target_batch, data=data)
                    st.session_state.file_idx += 1
                    st.rerun()
                else:
                    st.write(':red[Not Saved.]')

            with c2:
                pass
