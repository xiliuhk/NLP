__author__ = 'laceyliu'

import sys
import math
lambdas = sys.argv[1:5]
trainPath = sys.argv[6]
testPath = sys.argv[5]


def concatenate(path):
    f = open(path, 'r')
    content = f.read().replace('\n', '')
    f.close()
    return content

def buildVocab(text):
    vocabulary = {}
    tokens = filter(None, text.split(' '))
    for token in tokens:
        if vocabulary.has_key(token):
            vocabulary[token] += 1
        else:
            vocabulary[token] = 1
    vocabulary['UNKNOWNWORD'] = 0
    for key, value in vocabulary.items():
        if vocabulary[key] < 5:
            vocabulary['UNKNOWNWORD'] += 1
            while key in tokens:
                tokens.remove(key)
            del vocabulary[key]
    return vocabulary, len(tokens), tokens

def trainUnigram(vocab, cnt):
    unigrams = vocab.copy()
    for key, value in unigrams.items():
        unigrams[key] = value*1.0 / cnt

    return unigrams

def trainUniform(vocab):
    uniform = vocab.copy()
    size = len(vocab)
    for key, value in uniform.items():
        uniform[key] = 1.0 / size
    return uniform

def trainBigram(list, vocab):
    list.insert(0, 'START')
    bigrams = zip(list, list[1:])
    bigramsProb = {}
    for bigram in bigrams:
        if bigramsProb.has_key(bigram):
            bigramsProb[bigram] += 1
        else:
            bigramsProb[bigram] = 1
    bigrams = bigramsProb.copy()
    for key, value in bigramsProb.items():
        if 'START' in key:
            continue
        bigramsProb[key] = bigramsProb[key]*1.0/vocab[key[0]]
    return bigramsProb, bigrams

def trainTrigram(list, bigrams):
    list.insert(0, 'START')
    trigrams = zip(list, list[1:], list[2:])
    trigramsProb = {}
    for trigram in trigrams:
        if trigramsProb.has_key(trigram):
            trigramsProb[trigram] += 1
        else:
            trigramsProb[trigram] = 1
    for key, value in trigramsProb.items():
        if 'START' in key:
            continue
        trigramsProb[key] /=1.0* bigrams[key[1:]]
    return trigramsProb

def testBigram(list, model):
    s = 0.0
    n = 0
    for token in list:
        if n == 0:
            if ('START', token) in model:
                s += math.log(model[('START', token)])
        else:
            if (list[n-1], token) in model:
                s += math.log(model[(list[n-1], token)])
        n += 1
    return -s/n/math.log(2)

def testTrigram(list, model):
    s = 0.0
    n = 0
    for token in list:
        if n == 0:
            if ('START', 'START', token) in model:
                s += math.log(model[('START', 'START', token)])
        elif n == 1:
            if ('START', list[n-1], token) in model:
                s += math.log(model[('START', list[n-1], token)])
        else:
            if (list[n-2], list[n-1], token) in model:
                if model[(list[n-2], list[n-1], token)]<= 0:
                    print str((list[n-2], list[n-1], token)) + ' ' + model[(list[n-2], list[n-1], token)]
                s += math.log(model[(list[n-2], list[n-1], token)])
        n += 1
    return -s/n/math.log(2)

def testUnigram(list, model):
    s = 0.0
    n = 0
    for token in list:
        if token in model:
            s += math.log(model[token])
            n += 1
    return -s/n/math.log(2)

def testUniform(list, model):
    s = 0.0
    n = 0
    for token in list:
        if token in model:
            s += math.log(model[token])
            n += 1
    return -s/n/math.log(2)

def testInterpolated (list, um, bm, tm, uf, lambdas):

    h0 = float(lambdas[0])*testUniform(list, uf)
    h1 = float(lambdas[1])*testUnigram(list, um)
    h2 = float(lambdas[2])*testBigram(list, bm)
    h3 = float(lambdas[3])*testTrigram(list, tm)

    return math.pow(2, h0+h1+h2+h3)

if __name__ == "__main__":
    # train
    trainText = concatenate(trainPath)
    trainVocab, trainCnt, trainTokens = buildVocab(trainText)
    trainUnigrams = trainUnigram(trainVocab, trainCnt)
    trainUniforms = trainUniform(trainVocab)
    trainBigramProbs, trainBigrams = trainBigram(trainTokens, trainVocab)
    trainTrigramProbs = trainTrigram(trainTokens, trainBigrams)

    # test
    testText = concatenate(testPath)
    testVocab, testCnt, testTokens = buildVocab(testText)

    # Interpolated
    print testInterpolated(testTokens, trainUnigrams, trainBigramProbs,
                           trainTrigramProbs, trainUniforms, lambdas)

    # # subtask 1
    # print('===subtask 1.1')
    # print('unigram:'+str(len(trainUnigrams)))
    # print('bigram:'+str(len(trainBigramProbs)))
    # print('trigram:'+str(len(trainTrigramProbs)))
    #
    # print('===subtask 1.2')
    # sorted_unigram = sorted(trainUnigrams, key=trainUnigrams.get, reverse=True)
    # sorted_bigram = sorted(trainBigramProbs, key=trainBigramProbs.get, reverse=True)
    # sorted_trigram = sorted(trainTrigramProbs, key=trainTrigramProbs.get, reverse=True)
    # print(sorted_unigram[0])
    # print(sorted_bigram[0])
    # print(sorted_trigram[0])

    # print('subtask 1.3')
    # setNames = ['games', 'health', 'news', 'shopping', 'sports']
    # for name in setNames:
    #     trainText = concatenate('train/'+name+'.txt')
    #     trainVocab, trainCnt, trainTokens = buildVocab(trainText)
    #     trainBigramProbs, trainBigrams = trainBigram(trainTokens, trainVocab)
    #     trainTrigramProbs = trainTrigram(trainTokens, trainBigrams)
    #     sorted_bigram = sorted(trainBigramProbs, key=trainBigramProbs.get, reverse=True)
    #     sorted_trigram = sorted(trainTrigramProbs, key=trainTrigramProbs.get, reverse=True)
    #     print name
    #     print(sorted_bigram[0:10])
    #     print(sorted_trigram[0:10])

    # print('subtask 2')
    # setNames = ['games', 'health', 'news', 'shopping', 'sports']
    # for name in setNames:
    #     setNames2 = setNames[:]
    #     setNames2.remove(name)
    #     trainText = ''
    #     for n in setNames2:
    #         trainText += concatenate('train/'+n+'.txt')
    #     trainVocab, trainCnt, trainTokens = buildVocab(trainText)
    #     trainUnigrams = trainUnigram(trainVocab, trainCnt)
    #     trainUniforms = trainUniform(trainVocab)
    #     trainBigramProbs, trainBigrams = trainBigram(trainTokens, trainVocab)
    #     trainTrigramProbs = trainTrigram(trainTokens, trainBigrams)
    #     testText = concatenate('train/'+name+'.txt')
    #     testVocab, testCnt, testTokens = buildVocab(testText)
    #     print name
    #     print testInterpolated(testTokens, trainUnigrams, trainBigramProbs,
    #                        trainTrigramProbs, trainUniforms, lambdas)

    # print('subtask 3')
    # name = 'sports'
    # f = open('train/'+name+'.txt')
    # lines = f.readlines()
    # f.close()
    # trainText = ''.join(lines[:25]).replace('\n', '')
    # testText = ''.join(lines[25:]).replace('\n', '')
    #
    # # train
    # trainVocab, trainCnt, trainTokens = buildVocab(trainText)
    #
    # trainUnigrams = trainUnigram(trainVocab, trainCnt)
    # trainUniforms = trainUniform(trainVocab)
    # trainBigramProbs, trainBigrams = trainBigram(trainTokens, trainVocab)
    # trainTrigramProbs = trainTrigram(trainTokens, trainBigrams)
    #
    # # test
    # testVocab, testCnt, testTokens = buildVocab(testText)
    # print testInterpolated(testTokens, trainUnigrams, trainBigramProbs,
    #                        trainTrigramProbs, trainUniforms, lambdas)
    # # vice vera
    # # train
    # trainVocab, trainCnt, trainTokens = buildVocab(testText)
    #
    # trainUnigrams = trainUnigram(trainVocab, trainCnt)
    # trainUniforms = trainUniform(trainVocab)
    # trainBigramProbs, trainBigrams = trainBigram(trainTokens, trainVocab)
    # trainTrigramProbs = trainTrigram(trainTokens, trainBigrams)
    #
    # # test
    # testVocab, testCnt, testTokens = buildVocab(trainText)
    # print testInterpolated(testTokens, trainUnigrams, trainBigramProbs,
    #                        trainTrigramProbs, trainUniforms, lambdas)

    # # print ("bonus subtask 1")
    # setNames = ['games', 'health', 'news', 'shopping', 'sports']
    # name = 'shopping'
    # setNames2 = setNames[:]
    # setNames2.remove(name)
    # trainText = ''
    # for n in setNames2:
    #     trainText += concatenate('train/'+n+'.txt')
    # trainVocab, trainCnt, trainTokens = buildVocab(trainText)
    # trainUnigrams = trainUnigram(trainVocab, trainCnt)
    # trainUniforms = trainUniform(trainVocab)
    # trainBigramProbs, trainBigrams = trainBigram(trainTokens, trainVocab)
    # trainTrigramProbs = trainTrigram(trainTokens, trainBigrams)
    # testText = concatenate('train/'+name+'.txt')
    # testVocab, testCnt, testTokens = buildVocab(testText)
    # print name
    # print testInterpolated(testTokens, trainUnigrams, trainBigramProbs,
    #                    trainTrigramProbs, trainUniforms, lambdas)

    # print('bonus subtask 2')
    # name = 'shopping'
    # # train
    # trainText = concatenate('train/'+name+'.txt')
    # trainVocab, trainCnt, trainTokens = buildVocab(trainText)
    #
    # trainUnigrams = trainUnigram(trainVocab, trainCnt)
    # trainUniforms = trainUniform(trainVocab)
    # trainBigramProbs, trainBigrams = trainBigram(trainTokens, trainVocab)
    # trainTrigramProbs = trainTrigram(trainTokens, trainBigrams)
    #
    # # Interpolated
    # print testInterpolated(trainTokens, trainUnigrams, trainBigramProbs,
    #                        trainTrigramProbs, trainUniforms, lambdas)
