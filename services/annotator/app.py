import os
import pandas as pd
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import librosa.display


# set constants
CWD = os.getcwd()
# Define the path to the raw data folder of audio files to label
RAW_PATH = os.path.join(CWD, '../../data/raw')


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

    subdirs = os.listdir(RAW_PATH)

    # return if no unprocessed dirs
    if len(subdirs) < 1:
        return None

    unprocessed = []

    for batch in subdirs:
        if ".processed" in os.listdir(os.path.join(RAW_PATH, batch)):
            pass
        elif "annotations.json" in os.listdir(os.path.join(RAW_PATH, batch)):
            pass
        else:
            unprocessed.append(batch)

    return unprocessed


def fetch_unprocessed_batch(batch: str):
    batch_path = os.path.join(RAW_PATH, batch)
    files = os.listdir(batch_path)

    return [file for file in files if file.endswith(".wav")]


def append_annotation_file(batch: str, data):
    batch_path = os.path.join(RAW_PATH, batch)
    annotation_path = os.path.join(batch_path, 'annotations.json')
    # check if exists first, if not create
    if os.path.isfile(annotation_path):
        # TODO: append json
        pass
    else:
        # TODO: dump json into new file
        pass

def check_if_annotated(batch: str, filename: str):
    batch_path = os.path.join(RAW_PATH, batch)
    annotation_path = os.path.join(batch_path, 'annotations.json')
    try:
        # TODO: open annotations.json if filename error, assume none annotated yet
        pass
    except FileNotFoundError:
        return False
    finally:
        return False

unprocessed_batches = fetch_unprocessed_batches()
target_batch = unprocessed_batches[0]
target_batch_path = os.path.join(RAW_PATH, target_batch)
target_batch_files = fetch_unprocessed_batch(target_batch)

total_wav_files = len(target_batch_files)


# logic for selecting the next file not already in the output file
for file in target_batch_files:

    # check if the recording is in the annotations.json
    if not check_if_annotated(batch=target_batch, filename=file):

        filepath = os.path.join(target_batch_path, file)

        head1, head2 = st.columns(2)

        with head1:
            st.header(file)

            audio_file = open(filepath, 'rb')
            audio_bytes = audio_file.read()

            audible_range = st.slider('Ideal audio clip range', 0, 60, (0, 60), step=5)

            st.write(file)
            st.audio(audio_bytes, format='audio/wav')


        with head2:

            x, sr = librosa.load(filepath)

            fig, ax = plt.subplots()

            ax = librosa.display.waveshow(x)
            plt.yticks([])
            st.pyplot(fig)


            classification = st.checkbox('Aircraft audible', value=True)
            flag = st.checkbox('Flag clip', value=False)
            audible_start = audible_range[0]
            audible_end = audible_range[1]

            data = {
                "filename": file,
                "label": classification,
                "trim_start_s": audible_start,
                "trim_end": audible_end,
                "flag": flag
            }


            # commit results button
            c1, c2 = st.columns(2)

            with c1:
                sc1, sc2 = st.columns(2)

                with sc1:
                    # Submit button to commit the labelling data to the main CSV
                    if st.button('Commit'):
                        append_annotation_file(batch=target_batch, data=data)
                        st.empty()

                    else:
                        st.write(':red[Not Saved.]')


                with sc2:
                    if st.button('Next file'):
                        break

                    else:
                        pass

                    break

