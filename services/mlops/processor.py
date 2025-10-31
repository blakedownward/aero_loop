# this file is for trimming the audio files to the annotated range

import os
import scipy.io.wavfile.read


SAMPLE_RATE = 16000
DTYPE = 'int16'


def trim_wav(filepath: str, out_filepath: str, offset: int, duration: int):
    '''function to take a wav file, check it's sample rate and dtype, then write a file with given offset and duration'''
    sr, audio = scipy.io.wavfile.read(filepath)
    if sr == SAMPLE_RATE and audio.dtype == DTYPE:
        trim_audio = audio[(SAMPLE_RATE*offset):]
        trim_audio = trim_audio[:(SAMPLE_RATE*duration)]
        scipy.io.wavfile.write(out_filepath, SAMPLE_RATE, trim_audio)
    else:
        print('Sample rate or dtype mismatch: {filepath} skipped')