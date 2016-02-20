__author__ = 'laceyliu'

import os, string
paths = ['blue', 'red']

vocab = {}

insts = []

out = open('newTrain.txt', 'w')

for path in paths:
    dir = os.listdir('bonus_data/'+path+'/')
    for file in dir:
        if '.txt' not in file:
            continue
        speech = ''
        with open('bonus_data/'+path+'/'+file, 'r') as f:
            speech = f.read().upper()
        f.close()
        exclude = set(string.punctuation)
        speech = ''.join(ch for ch in speech if ch not in exclude)
        speech = speech.replace('\n', '').replace('\r', '')
        wlist = speech.split(' ')
        for word in wlist:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
        insts.append((path, wlist))

for inst in insts:
    out.write(inst[0].upper() + '\t')
    for w in inst[1]:
        if vocab[w] == 1:
            inst[1].remove(w)
    for w in inst[1]:
        out.write(w + ' ')
    out.write('\n')

out.close()



