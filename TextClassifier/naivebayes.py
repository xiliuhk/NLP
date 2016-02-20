__author__ = 'laceyliu'

import sys
import math
from collections import defaultdict


def createDataset(path):
    ret = []
    with open(path, 'r') as f:
        for line in f:
            t = line.split('\t')
            # list = t[1].split(' ')
            # ret.append((t[0], zip(list, list[1:])))
            ret.append((t[0], t[1].split(' ')))
            # ret.append((t[0], removeStopwords(t[1].split(' '))))
    f.close()
    return ret

def train(trainSet):
    global vocab, tokenMap, p_class, pMap, lenMap
    vocab = {}
    p_class = {'RED':0.0, 'BLUE': 0.0}
    lenMap = {'RED':0, 'BLUE': 0}
    tokenMap = defaultdict(dict)
    pMap = defaultdict(dict)

    # count tokens by class
    for inst in trainSet:
        # p(BLUE), p(RED)
        p_class[inst[0]] += 1
        for token in inst[1]:
            if token not in tokenMap[inst[0]].keys():
                tokenMap[inst[0]][token] = 1
            else:
                tokenMap[inst[0]][token] += 1
            if token not in vocab.keys():
                vocab[token] = 1
            else:
                vocab[token] += 1
            lenMap[inst[0]] += 1

    p_class['BLUE'] = math.log(p_class['BLUE'])
    p_class['RED'] = math.log(p_class['RED'])
    norm = len(vocab)

    for color, cntMap in tokenMap.items():
        for token, cnt in cntMap.items():
            pMap[token][color] = math.log((cnt+1.0)/(norm+lenMap[color]))

def test(testSet):
    global vocab, tokenMap, p_class, pMap, lenMap

    hit_blue = 0
    hit_red = 0
    act_blue = 0
    pred_blue = 0

    for inst in testSet:
        p_blue = p_class['BLUE']
        p_red = p_class['RED']
        if inst[0] == 'BLUE':
            act_blue += 1
        for token in inst[1]:
            if 'BLUE' not in pMap[token].keys():
                p_blue += math.log(1.0/len(vocab)+lenMap['BLUE'])
            else:
                p_blue += pMap[token]['BLUE']
            if 'RED' not in pMap[token].keys():
                p_red += math.log(1.0/len(vocab)+lenMap['RED'])
            else:
                p_red += pMap[token]['RED']
            pred = ''

        if p_blue < p_red:
            pred = 'BLUE'
            pred_blue += 1
        else:
            pred = 'RED'

        if inst[0] == pred:
            if pred == 'BLUE':
                hit_blue += 1
            else:
                hit_red += 1
        pb = p_blue/(p_blue+p_red)
        pr = p_red/(p_blue+p_red)

        sys.stdout.write(pred + '-->' + inst[0]+ '| Blue: red ' + str(pb)+' : '+str(pr)+'\n')
    total = len(testSet)

    accuracy = 1.0*(hit_blue + hit_red)/total
    recall_blue = 1.0*hit_blue/act_blue
    recall_red = 1.0*hit_red/(total-act_blue)
    precision_blue = 1.0*hit_blue/pred_blue
    precision_red = 1.0*hit_red/(total-pred_blue)

    sys.stdout.write('Overall Accuracy: ' + str(accuracy) + '\n')
    sys.stdout.write('Precision (BLUE): ' + str(precision_blue) + '\n')
    sys.stdout.write('Recall (BLUE): ' + str(recall_blue) + '\n')
    sys.stdout.write('Precision (RED): ' + str(precision_red) + '\n')
    sys.stdout.write('Recall (RED): ' + str(recall_red) + '\n')

def removeStopwords(list):
    stopwords = []
    with open('stopwords.txt') as sf:
        stopwords = sf.read().splitlines()
    sf.close()
    stopwords.append('UNKNOWNWORD')
    for sw in stopwords:
        if sw in list:
            list.remove(sw)
    return list


if __name__ == "__main__":
    trainSet = createDataset(sys.argv[1])
    train(trainSet)
    testSet = createDataset(sys.argv[2])
    test(testSet)


