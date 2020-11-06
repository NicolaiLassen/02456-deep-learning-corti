import soundfile as sf


# ref: https://github.com/mailong25/vietnamese-speech-recognition/blob/master/wav2vec.py
def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """
    wav, sr = sf.read(fname)
    assert sr == 16e3
    return wav, 16e3


