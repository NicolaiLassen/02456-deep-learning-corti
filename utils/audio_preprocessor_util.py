import os
from multiprocessing import Pool

import librosa
import soundfile as sf
from torch.utils.data.dataloader import DataLoader

from audio_lib.audio_segmentation import segment_audio


def convert_file_format(in_path, out_path):
    command = f'ffmpeg -i \"{in_path}\" {out_path}'
    os.system(command)


# ref: https://github.com/mailong25/vietnamese-speech-recognition/blob/master/wav2vec.py
def convert_to_16k(in_path, out_path):
    _, file_extension = os.path.splitext(in_path)
    y, s = librosa.load(in_path, sr=8000)
    y_16k = librosa.resample(y, s, 16000)
    sf.write(out_path, y_16k, 16000, format=file_extension)


# ref: https://github.com/mailong25/vietnamese-speech-recognition/blob/master/wav2vec.py
def stereo2mono(x):
    # (stored in a numpy array) to MONO (if it is STEREO)
    if isinstance(x, int):
        return -1
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        if x.shape[1] == 1:
            return x.flatten()
        else:
            if x.shape[1] == 2:
                return ((x[:, 1] / 2) + (x[:, 0] / 2))
            else:
                return -1


# ref: https://github.com/mailong25/vietnamese-speech-recognition/blob/master/wav2vec.py
def preprocessing(args):
    # args: [file_path, file_index, output_path]
    file_path = args[0]
    file_index = args[1]
    output_path = args[2]

    _, file_extension = os.path.splitext(file_path)
    new_file_path = os.path.join(output_path, str(file_index) + file_extension)
    convert_to_16k(file_path, new_file_path)

    ## TODO
    segment_audio(new_file_path, output_path, max_len=12)


class AudioPreprocessor:
    def __init__(self, n_thread=2, out_path='./temp'):
        self.n_thread = n_thread
        self.output_path = os.path.abspath(out_path)
        self.pool = Pool(self.n_thread)

    def transcribe(self, sound_files):
        self.pool.map(preprocessing, [(sound_files[i], i, self.output_path) for i in range(0, len(sound_files))])
        self.pool.terminate()

    def load_data(self, batch_size=256, test_size=0.1, valid_size=0.2):
        # Check if the file path is created for our preprocessed data
        assert os.path.isdir(self.output_path) == True

        data_set = DataLoader(self.output_path,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=self.n_thread)

        # waveform, sample_rate = torchaudio.load(filename)
