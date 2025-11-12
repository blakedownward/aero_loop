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
    # Initialize session state for workflow steps
    if 'workflow_step' not in st.session_state:
        st.session_state.workflow_step = 'process'
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    if 'upload_done' not in st.session_state:
        st.session_state.upload_done = False
    if 'training_done' not in st.session_state:
        st.session_state.training_done = False
    if 'evaluation_result' not in st.session_state:
        st.session_state.evaluation_result = None
    if 'build_done' not in st.session_state:
        st.session_state.build_done = False
    
    st.success("All batches annotated. Ready to process.")
    
    # Step 1: Process Data
    if st.session_state.workflow_step == 'process':
        if st.button("Process Data"):
            with st.spinner("Processing annotated batches..."):
                ok, msg = _run_processor()
            if ok:
                st.success("Processing complete. Outputs in data/processed.")
                _show_processing_logs()
                st.session_state.processing_done = True
                st.session_state.workflow_step = 'upload'
                st.rerun()
            else:
                st.error(f"Processing failed: {msg}")
    
    # Step 2: Upload and Train
    elif st.session_state.workflow_step == 'upload':
        if st.session_state.processing_done:
            st.info("Processing complete. Ready to upload to Edge Impulse and train.")
            if st.button("Upload to Edge Impulse & Train Model"):
                with st.spinner("Uploading samples and starting training..."):
                    result = _run_upload_and_train()
                if result.get('success'):
                    st.success("Training started successfully!")
                    st.session_state.upload_done = True
                    st.session_state.training_done = True
                    st.session_state.workflow_step = 'evaluate'
                    st.rerun()
                else:
                    st.error(f"Upload/Train failed: {result.get('error', 'Unknown error')}")
    
    # Step 3: Evaluate Model
    elif st.session_state.workflow_step == 'evaluate':
        if st.session_state.training_done:
            st.info("Training complete. Evaluating model performance...")
            with st.spinner("Fetching evaluation metrics..."):
                eval_result = _run_evaluate()
            
            if eval_result.get('success'):
                st.session_state.evaluation_result = eval_result
                _show_evaluation_results(eval_result)
                
                if eval_result.get('should_build', False):
                    st.session_state.workflow_step = 'build'
                    st.rerun()
                else:
                    st.warning("Model did not improve. No deployment needed.")
                    if st.button("Start Over"):
                        _reset_workflow()
            else:
                st.error(f"Evaluation failed: {eval_result.get('error', 'Unknown error')}")
    
    # Step 4: Build and Download (if improved)
    elif st.session_state.workflow_step == 'build':
        if st.session_state.evaluation_result and st.session_state.evaluation_result.get('should_build'):
            st.info("Model improved! Building deployment...")
            if st.button("Build & Download Model"):
                with st.spinner("Building deployment and downloading model..."):
                    build_result = _run_build_and_download()
                if build_result.get('success'):
                    st.success("Model built and downloaded successfully!")
                    st.session_state.build_done = True
                    st.session_state.workflow_step = 'deploy'
                    st.rerun()
                else:
                    st.error(f"Build/Download failed: {build_result.get('error', 'Unknown error')}")
    
    # Step 5: Deploy to Pi (if improved)
    elif st.session_state.workflow_step == 'deploy':
        if st.session_state.build_done:
            st.info("Model ready for deployment. Deploy to Raspberry Pi?")
            if st.button("Deploy to Pi & Reboot"):
                with st.spinner("Deploying model to Pi..."):
                    deploy_result = _run_deploy()
                if deploy_result.get('success'):
                    st.success("Model deployed successfully! Pi will reboot shortly.")
                    if st.button("Start Over"):
                        _reset_workflow()
                else:
                    st.error(f"Deployment failed: {deploy_result.get('error', 'Unknown error')}")


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


def _reset_workflow():
    """Reset workflow state to start over."""
    st.session_state.workflow_step = 'process'
    st.session_state.processing_done = False
    st.session_state.upload_done = False
    st.session_state.training_done = False
    st.session_state.evaluation_result = None
    st.session_state.build_done = False
    st.rerun()


def _run_upload_and_train():
    """Upload samples to Edge Impulse and start training."""
    try:
        # Add repo root to path for imports
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        
        from services.mlops.ei_uploader import upload_processed_samples
        from services.mlops.ei_trainer import start_training, wait_for_training
        
        # Upload samples
        upload_result = upload_processed_samples(progress_callback=lambda msg: None)
        if not upload_result.get('success'):
            return {'success': False, 'error': f"Upload failed: {upload_result.get('error')}"}
        
        # Start training
        train_result = start_training(progress_callback=lambda msg: None)
        if not train_result.get('success'):
            return {'success': False, 'error': f"Training start failed: {train_result.get('error')}"}
        
        job_id = train_result.get('job_id')
        
        # Wait for training to complete
        wait_result = wait_for_training(
            job_id=job_id,
            progress_callback=lambda msg: None
        )
        
        if wait_result.get('success') and wait_result.get('status') == 'completed':
            return {'success': True, 'job_id': job_id}
        else:
            error_msg = wait_result.get('error', f"Training status: {wait_result.get('status', 'unknown')}")
            return {'success': False, 'error': f"Training failed: {error_msg}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _run_evaluate():
    """Evaluate model and compare with best model."""
    try:
        # Add repo root to path for imports
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        
        from services.mlops.ei_downloader import get_model_metrics
        from services.mlops.model_evaluator import evaluate_model
        
        # Get evaluation metrics
        model_metrics = get_model_metrics()
        if not model_metrics:
            return {'success': False, 'error': 'Could not retrieve model metrics'}
        
        # Evaluate against best model
        eval_result = evaluate_model(model_metrics)
        
        return {
            'success': True,
            'evaluation': eval_result,
            'model_metrics': model_metrics,
            'should_build': eval_result.get('is_better', False),
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _show_evaluation_results(eval_result):
    """Display evaluation results and comparison."""
    eval_data = eval_result.get('evaluation', {})
    new_metrics = eval_data.get('new_metrics', {})
    best_metrics = eval_data.get('best_metrics', {})
    is_better = eval_data.get('is_better', False)
    reason = eval_data.get('reason', '')
    
    st.subheader("Model Evaluation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Model**")
        st.metric("Accuracy", f"{new_metrics.get('accuracy', 0):.2f}%")
        if new_metrics.get('loss') != float('inf'):
            st.metric("Loss", f"{new_metrics.get('loss', 0):.4f}")
    
    with col2:
        st.write("**Best Model**")
        if best_metrics:
            st.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.2f}%")
            if best_metrics.get('loss') != float('inf'):
                st.metric("Loss", f"{best_metrics.get('loss', 0):.4f}")
        else:
            st.write("(First model)")
    
    if is_better:
        st.success(f"✓ {reason}")
    else:
        st.warning(f"✗ {reason}")


def _run_build_and_download():
    """Build deployment and download model."""
    try:
        # Add repo root to path for imports
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        
        from services.mlops.ei_downloader import build_deployment, wait_for_build_completion, download_model
        
        # Build deployment
        build_result = build_deployment(progress_callback=lambda msg: None)
        if not build_result.get('success'):
            return {'success': False, 'error': f"Build failed: {build_result.get('error')}"}
        
        job_id = build_result.get('job_id')
        
        # Wait for build
        build_status = wait_for_build_completion(job_id, progress_callback=lambda msg: None)
        if build_status.get('status') != 'completed':
            return {'success': False, 'error': f"Build did not complete: {build_status.get('status')}"}
        
        # Download model
        download_result = download_model(progress_callback=lambda msg: None)
        if not download_result.get('success'):
            return {'success': False, 'error': f"Download failed: {download_result.get('error')}"}
        
        return {'success': True, 'model_filename': download_result.get('model_filename')}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _run_deploy():
    """Deploy model to Raspberry Pi."""
    try:
        # Add repo root to path for imports
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        
        from services.remote.deploy_model import deploy_latest_model
        
        result = deploy_latest_model(progress_callback=lambda msg: None)
        return result
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _show_processing_logs():
    """Display latest processing run summary and recent label rows."""
    proc_dir = os.path.join(REPO_ROOT, 'data', 'processed')
    run_log_csv = os.path.join(proc_dir, 'run_log.csv')
    labels_csv = os.path.join(proc_dir, 'labels.csv')

    st.subheader("Processing Summary")
    if os.path.isfile(run_log_csv):
        try:
            df = pd.read_csv(run_log_csv)
            if not df.empty:
                last = df.tail(1).reset_index(drop=True)
                # Coerce seconds to int for display
                sec_cols = [
                    'aircraft_pre_s','aircraft_post_s','negative_pre_s','negative_post_s'
                ]
                for c in sec_cols:
                    if c in last.columns:
                        last[c] = pd.to_numeric(last[c], errors='coerce').fillna(0).astype(int)

                # Compute class counts for this run from labels.csv tail(processed_count)
                if os.path.isfile(labels_csv) and 'processed_count' in last.columns:
                    try:
                        n = int(pd.to_numeric(last.loc[0, 'processed_count'], errors='coerce'))
                    except Exception:
                        n = 0
                    if n > 0:
                        try:
                            dfl = pd.read_csv(labels_csv)
                            tail = dfl.tail(n)
                            ac = int((tail['class'] == 'aircraft').sum()) if 'class' in tail.columns else 0
                            nc = int((tail['class'] == 'negative').sum()) if 'class' in tail.columns else 0
                            last['aircraft_count'] = ac
                            last['negative_count'] = nc
                        except Exception:
                            # Ignore counting errors; just skip counts
                            pass

                cols = [
                    'run_timestamp','batches_count','annotated_count','processed_count',
                    'aircraft_pre_s','aircraft_post_s','negative_pre_s','negative_post_s',
                    'aircraft_count','negative_count'
                ]
                present = [c for c in cols if c in last.columns]
                st.table(last[present])
            else:
                st.info("No summary rows found in run_log.csv yet.")
        except Exception as e:
            st.warning(f"Could not read run_log.csv: {e}")
    else:
        st.info("run_log.csv not found yet.")


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
