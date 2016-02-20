__author__ = 'laceyliu'
import sys
import math
f = open(sys.argv[1], 'r')
test = f.read().splitlines()
f.close()

f = open(sys.argv[2], 'r')
gold = f.read().splitlines()
f.close()

if not len(gold) == len(test):
    sys.stdout.write("Invalid files \n")
    exit()

avg = 0.0
mx = 0.0
mn = 1.0
for i in xrange(0, len(gold)):
    test_tags = test[i].split(" ")
    gold_tags = gold[i].split(" ")
    err = 0
    for j in xrange(0, len(gold_tags)):
        err += (not test_tags[j] == gold_tags[j])
    p = 1-1.0*err/len(gold_tags)
    avg += p/len(gold)
    mx = max(mx, p)
    mn = min(mn, p)

sys.stdout.write("Avg: "+ str(avg)+"\nMax: "+str(mx)+"\nMin: "+str(mn)+"\n")