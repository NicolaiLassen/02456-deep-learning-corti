import ntpath
import os
from shutil import copy2

from pydub import AudioSegment


###
# ref: https://github.com/mailong25/vietnamese-speech-recognition/blob/master/wav2vec.py
###

def chunk_audio(file_path, output_path, max_len=12):
    file_name = ntpath.basename(file_path)
    audio = AudioSegment.from_wav(file_path)

    if len(audio) / 1000 > max_len:
        segs = silenceRemovalWrapper(file_path)
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


def silenceRemovalWrapper(inputFile, smoothingWindow=0.5, weight=0.2, saveFile=False):
    if not os.path.isfile(inputFile):
        raise Exception("Input audio_lib file not found!")
    file = inputFile.split('/')[-1].split('.')[0]
    [fs, x] = audioBasicIO.readAudioFile(inputFile)
    segmentLimits = aS.silenceRemoval(x, fs, 0.03, 0.03,
                                      smoothingWindow, weight, False)
    for i, s in enumerate(segmentLimits):
        strOut = "{0:s}_{1:.2f}-{2:.2f}.wav".format(inputFile[0:-4], s[0], s[1])
        if saveFile:
            file.write(strOut, fs, x[int(fs * s[0]):int(fs * s[1])])

    return segmentLimits
