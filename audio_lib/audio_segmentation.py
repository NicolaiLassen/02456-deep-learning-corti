import ntpath
import os
from shutil import copy2

import numpy
from pydub import AudioSegment

from audio_lib.audio_io import readAudioFile
from audio_lib.audio_silence_removal import silenceRemoval


def smoothMovingAvg(inputSignal, windowLen=11):
    windowLen = int(windowLen)
    if inputSignal.ndim != 1:
        raise ValueError("")
    if inputSignal.size < windowLen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if windowLen < 3:
        return inputSignal
    s = numpy.r_[2 * inputSignal[0] - inputSignal[windowLen - 1::-1],
                 inputSignal, 2 * inputSignal[-1] - inputSignal[-1:-windowLen:-1]]
    w = numpy.ones(windowLen, 'd')
    y = numpy.convolve(w / w.sum(), s, mode='same')
    return y[windowLen:-windowLen + 1]


def segment_audio(file_path, output_path, max_len=12):
    '''

    :param file_path:
    :param output_path:
    :param max_len:
    :return:
    '''

    file_name = ntpath.basename(file_path)
    audio = AudioSegment.from_wav(file_path)

    if len(audio) / 1000 > max_len:
        segs = silence_removal_segment_wrapper(file_path)
        segs = sorted([j for i in segs for j in i])
        points = []
        total = 0

        for i in range(1, len(segs)):
            gap = segs[i] - segs[i - 1]
            if total + gap > max_len:
                points.append(segs[i - 1])
                total = gap
                while (total > max_len):
                    points.append(points[-1] + max_len)
                    total -= max_len
            else:
                total += gap

        points.append(segs[-1])

        if points[0] != segs[0]:
            points.insert(0, segs[0])
        points = [int(p * 1000) for p in points]

        for i in range(1, len(points)):
            part = audio[points[i - 1]:points[i]]
            path_to_write = os.path.join(output_path, file_name.replace('.wav', '_' + str(i - 1) + '.wav'))
            part.export(path_to_write, format='wav')
    else:
        path_to_write = os.path.join(output_path, file_name.replace('.wav', '_0.wav'))
        copy2(file_path, path_to_write)

    if output_path in file_path:
        os.remove(file_path)


def silence_removal_segment_wrapper(inputFile, smoothingWindow=0.5, weight=0.2, saveFile=False):
    if not os.path.isfile(inputFile):
        raise Exception("Input audio_lib file not found!")

    file = inputFile.split('/')[-1].split('.')[0]
    [fs, x] = readAudioFile(inputFile)
    segment_limits = silenceRemoval(x, fs, 0.03, 0.03, smoothingWindow, weight, False)

    for i, s in enumerate(segment_limits):
        strOut = "{0:s}_{1:.2f}-{2:.2f}.wav".format(inputFile[0:-4], s[0], s[1])
        if saveFile:
            file.write(strOut, fs, x[int(fs * s[0]):int(fs * s[1])])

    return segment_limits
