###
# ref: https://github.com/mailong25/vietnamese-speech-recognition/blob/master/wav2vec.py
###

def read_result(path):
    trans = {}

    with open(path) as f:
        data = f.read().split('\n')
        data = data[:-1]

    for d in data:
        end = d.find('(')
        pred = d[0:end - 1]
        file_name = d[end:].replace('(', '').replace(')', '')
        index = int(file_name.split('_')[-1].replace('.wav', ''))
        base_name = int(file_name.split('_')[0])

        if base_name not in trans:
            trans[base_name] = [(index, pred)]
        else:
            trans[base_name].append((index, pred))

    for name in trans:
        trans[name] = sorted(trans[name], key=lambda x: x[0])
        trans[name] = ' '.join([t[1] for t in trans[name]])

    return trans
