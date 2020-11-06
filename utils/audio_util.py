import librosa
import soundfile as sf


def convert_to_16k(in_path, out_path, format):
    y, s = librosa.load(in_path, sr=8000)
    y_16k = librosa.resample(y, s, 16000)
    sf.write(out_path, y_16k, 16000, format=format)


def chunk_audio():
