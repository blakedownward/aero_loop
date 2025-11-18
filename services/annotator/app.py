import os
import json
import wave
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


def get_inference_log_stats(batch: str):
    """Get summary statistics from inference_log.jsonl for a batch.
    Returns dict with 'total', 'saved', 'deleted', 'aircraft', 'negative', 
    'negative_saved', 'negative_deleted', 'negative_rejection_pct' counts, or None if file doesn't exist.
    """
    batch_path = os.path.join(RAW_PATH, batch)
    inference_log_path = os.path.join(batch_path, 'inference_log.jsonl')
    
    if not os.path.isfile(inference_log_path):
        return None
    
    stats = {
        'total': 0,
        'saved': 0,
        'deleted': 0,
        'aircraft': 0,
        'negative': 0,
        'negative_saved': 0,
        'negative_deleted': 0,
        'negative_rejection_pct': 0.0
    }
    
    # Load annotations to determine labels (most accurate)
    annotations = _load_annotations(batch)
    filename_to_label = {ann.get("filename"): ann.get("label") for ann in annotations if ann.get("filename")}
    
    # Count all entries using their status field
    try:
        with open(inference_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    status = entry.get("status", "unknown")
                    stats['total'] += 1
                    filename = entry.get("file")
                    max_pred = entry.get("max_pred")
                    
                    # Determine if this entry is negative or aircraft
                    # Priority: 1) annotation, 2) filename prefix (000000 = negative), 3) prediction
                    label = filename_to_label.get(filename) if filename else None
                    is_negative = False
                    is_aircraft = False
                    
                    if label is True:
                        is_aircraft = True
                    elif label is False:
                        is_negative = True
                    elif label is None:
                        # No annotation, check filename prefix first
                        if filename and filename.startswith("000000"):
                            is_negative = True
                        elif max_pred is not None:
                            # Fall back to prediction
                            if max_pred < 0.5:
                                is_negative = True
                            else:
                                is_aircraft = True
                    
                    # Count based on status field
                    if status == "saved":
                        stats['saved'] += 1
                        if is_aircraft:
                            stats['aircraft'] += 1
                        elif is_negative:
                            stats['negative'] += 1
                            stats['negative_saved'] += 1
                    elif status == "deleted":
                        stats['deleted'] += 1
                        if is_negative:
                            stats['negative_deleted'] += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        return None
    
    # Calculate rejection percentage for negative samples
    total_negative = stats['negative_saved'] + stats['negative_deleted']
    if total_negative > 0:
        stats['negative_rejection_pct'] = round((stats['negative_deleted'] / total_negative) * 100, 1)
    
    return stats


def load_predictions(batch: str, filename: str):
    """Load predictions array from inference_log.jsonl for the given filename.
    Only considers entries with "status": "saved". If multiple saved entries exist,
    returns the most recent one (last one found in file).
    Returns tuple of (predictions_list, num_periods) or None if not found.
    """
    batch_path = os.path.join(RAW_PATH, batch)
    inference_log_path = os.path.join(batch_path, 'inference_log.jsonl')
    
    if not os.path.isfile(inference_log_path):
        return None
    
    saved_entry = None
    try:
        with open(inference_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Check if this entry matches the filename and has "saved" status
                    if entry.get("file") == filename and entry.get("status") == "saved":
                        predictions = entry.get("predictions")
                        if predictions is not None:
                            # Keep the most recent saved entry (last one found)
                            saved_entry = entry
                except json.JSONDecodeError:
                    continue
        
        # Return the saved entry if found
        if saved_entry is not None:
            predictions = saved_entry.get("predictions")
            num_periods = saved_entry.get("num_periods", len(predictions) if predictions else 0)
            if predictions is not None:
                # Round predictions to 2 decimals
                rounded_preds = [round(float(p), 2) for p in predictions]
                return (rounded_preds, num_periods)
    except Exception:
        pass
    
    return None

unprocessed_batches = fetch_unprocessed_batches()


def _render_all_done():
    """Render UI when all batches are annotated."""
    st.success("✓ All batches annotated!")
    st.info("Use `run_train_deploy.bat` (or `run_train_deploy.sh`) to process, train, and deploy.")


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
            # Display inference log statistics
            inference_stats = get_inference_log_stats(batch=target_batch)
            if inference_stats:
                stats_text = f"Inference Log: {inference_stats['total']} total samples ({inference_stats['saved']} saved, {inference_stats['deleted']} deleted)"
                if inference_stats['saved'] > 0:
                    stats_text += f" | Saved: {inference_stats['aircraft']} aircraft, {inference_stats['negative']} negative"
                st.caption(stats_text)
                total_negative = inference_stats['negative_saved'] + inference_stats['negative_deleted']
                if total_negative > 0:
                    st.caption(f"Negative rejection rate: {inference_stats['negative_rejection_pct']}%")
            
            # Load and display predictions if available
            pred_data = load_predictions(batch=target_batch, filename=file)
            if pred_data is not None:
                predictions, num_periods = pred_data
                pred_str = "[" + ", ".join(str(p) for p in predictions) + "]"
                # st.write(pred_str)
            
            x, sr = librosa.load(filepath, sr=None)
            fig, ax = plt.subplots()
            librosa.display.waveshow(x, sr=sr, ax=ax)
            ax.set_yticks([])
            
            # Overlay predictions if available
            if pred_data is not None:
                predictions, num_periods = pred_data
                duration = len(x) / sr  # audio duration in seconds
                # Calculate time points for each prediction (centered in each segment)
                segment_duration = duration / len(predictions)
                time_points = [i * segment_duration + segment_duration / 2 for i in range(len(predictions))]
                
                # Create secondary y-axis for predictions
                ax2 = ax.twinx()
                ax2.plot(time_points, predictions, 'r-', linewidth=2, alpha=0.7, label='Predictions')
                ax2.set_ylabel('Prediction Probability', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                ax2.set_ylim([0, 1])
                ax2.grid(True, alpha=0.3)
            
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
