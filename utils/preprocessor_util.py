import os
from multiprocessing import Pool

import librosa
import soundfile as sf
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from audio_lib.audio_segmentation import segment_audio


###
# ref: https://github.com/mailong25/vietnamese-speech-recognition/blob/master/wav2vec.py
###


def convert_to_16k(in_path, out_path, format):
    y, s = librosa.load(in_path, sr=8000)
    y_16k = librosa.resample(y, s, 16000)
    sf.write(out_path, y_16k, 16000, format=format)


def preprocessing(args):
    '''
    Thread job wrapper for segmentation
    :param args:
    :return:
    '''
    file_path = args[0]
    file_index = args[1]
    output_path = args[2]
    new_file_path = os.path.join(output_path, str(file_index) + '.wav')
    convert_to_16k(file_path, new_file_path, "wav")
    segment_audio(new_file_path, output_path, max_len=12)


class AudioPreprocessor:
    '''
    '''
    def __init__(self,
                 valid_prefix="valid",
                 train_prefix="train",
                 in_path='.data/train-clean-100',
                 n_thread_decoder=2,
                 out_path='./temp'):
        self.output_path = os.path.abspath(out_path)
        self.pool = Pool(n_thread_decoder)

    def transcribe(self, sound_files):
        '''
        
        :param sound_files:
        :return:
        '''
        self.pool.map(preprocessing, [(sound_files[i], i, self.output_path) for i in range(0, len(sound_files))])

    def load_data(self,
                  threads=2,
                  batch_size=16,
                  test_size=0.0,
                  valid_size=0.2
                  ):
        # Check if the file path is created for our preprocessed data
        assert os.path.isdir(self.output_path) == True

        data_set = DataLoader(self.output_path,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=threads)

        train_sampler = SubsetRandomSampler(train_new_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(data_set, batch_size=batch_size,
                                  sampler=train_sampler, num_workers=1)
        valid_loader = DataLoader(data_set, batch_size=batch_size,
                                  sampler=valid_sampler, num_workers=1)
