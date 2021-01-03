import os

import librosa
import soundfile as sf


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
