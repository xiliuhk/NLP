__author__ = 'laceyliu'
# Python script to plot error by sentence and error by word
# 1. Slice training set into 20 folds
# 2. Select 1, 2, 3, ...20 folds to train HMM
# 3. Test with ptb.22.txt
# 4. Evaluate with ptb.22.txt
# Note:
# 1. matplotlib should be installed before generating the plot
# 2. this script takes about 1000 secs to because it has to wait for I/O before evaluation

import sys, subprocess
import matplotlib.pyplot as plt
import time
DATA = "data/sub"

tag_f = open("ptb.2-21.tgs",'r')
tags = tag_f.read().splitlines()
tag_f.close()
txt_f = open("ptb.2-21.txt", 'r')
text = txt_f.read().splitlines()
txt_f.close()

tlen = len(tags)
telen = len(text)
if not tlen == telen:
    sys.stdout.write("Invalid files")
    exit()

tlen /= 20

errs_word = []
errs_sent = []

for i in xrange(1, 21):
    tg = DATA+str(i)+'.tgs'
    tx = DATA+str(i)+'.txt'
    hmm = DATA+str(i)+'.hmm'
    out = DATA+str(i)+'.out'
    tgf = open(tg, 'w')
    txf = open(tx, 'w')
    t_tags = tags[:i*tlen]
    t_txts = text[:i*tlen]
    for j in xrange(0, i*tlen):
        tgf.write(t_tags[j]+'\n')
        txf.write(t_txts[j]+'\n')
    tgf.close()
    txf.close()
    cmd = "./train_hmm.py " + tg + " " +tx+" > "+hmm
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    time.sleep(i*3)
    cmd = "./viterbi.pl " + hmm + " < ptb.22.txt > "+out
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    time.sleep(i*5)
    cmd = "./tag_acc.pl ptb.22.tgs " + out
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    errs = proc.stdout.read().splitlines()
    err_w = float(errs[0].split(':')[1].split("(")[0].replace(" ", ""))
    err_s = float(errs[1].split(':')[1].split("(")[0].replace(" ", ""))
    errs_word.append(err_w)
    errs_sent.append(err_s)

x = range(20)
plt.plot(x, errs_word, 'r--', x, errs_sent, 'g^')
plt.show()










