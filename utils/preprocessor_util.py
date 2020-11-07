import os
from multiprocessing import Pool

import librosa
import soundfile as sf

from audio_lib.audio_segmentation import chunk_audio


###
# ref: https://github.com/mailong25/vietnamese-speech-recognition/blob/master/wav2vec.py
###


def convert_to_16k(in_path, out_path, format):
    y, s = librosa.load(in_path, sr=8000)
    y_16k = librosa.resample(y, s, 16000)
    sf.write(out_path, y_16k, 16000, format=format)


def preprocessing(args):
    file_path = args[0]
    file_index = args[1]
    output_path = args[2]
    new_file_path = os.path.join(output_path, str(file_index) + '.wav')
    convert_to_16k(file_path, new_file_path, "wav")
    chunk_audio(new_file_path, output_path, max_len=12)


class Preprocessor:
    def __init__(self,
                 w2letter,
                 w2vec,
                 am,
                 tokens,
                 lexicon,
                 lm,
                 nthread_decoder=4,
                 lmweight=1.51,
                 wordscore=2.57,
                 beamsize=200,
                 temp_path='./temp'):
        self.output_path = os.path.abspath(temp_path)
        self.pool = Pool(nthread_decoder)

    def transcribe(self, wav_files):
        preprocessing()

        self.pool.map(preprocessing, [(wav_files[i], i, self.output_path) for i in range(0, len(wav_files))])
