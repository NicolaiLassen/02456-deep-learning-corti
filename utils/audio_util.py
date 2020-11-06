import librosa
import soundfile as sf

# ref: https://github.com/mailong25/vietnamese-speech-recognition/blob/master/wav2vec.py
def convert_to_16k(in_path, out_path, format):
    y, s = librosa.load(in_path, sr=8000)
    y_16k = librosa.resample(y, s, 16000)
    sf.write(out_path, y_16k, 16000, format=format)

# ref: https://github.com/mailong25/vietnamese-speech-recognition/blob/master/wav2vec.py
def chunk_audio():
