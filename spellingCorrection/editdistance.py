__author__ = 'laceyliu'

import sys
import time

dictionary = []
raw = []
trie = None

class TrieNode:
    def __init__(self):
        self.word = None
        self.children = {}
    def insert(self, word):
        node = self
        for l in word:
            if l not in node.children:
                node.children[l] = TrieNode()
            node = node.children[l]
        node.word = word


def Levenshtein(t, w):
    cols = len(t) + 1
    rows = len(w) + 1

    # first row
    curRow = [0]
    for col in xrange(1, cols):
        curRow.append(curRow[col-1] + 1)

    # calculate dists
    for row in xrange(1, rows):
        prevRow = curRow
        curRow = [prevRow[0] + 1]
        for col in xrange(1, cols):
            insert = curRow[col-1] + 1
            delete = prevRow[col] + 1
            replace = prevRow[col-1] + (t[col-1] != w[row-1])
            curRow.append(min(insert, delete, replace))
    return curRow[-1]

def search(w, mode):
    minCost = sys.maxint
    target = ''
    if mode == 1:
        for t in dictionary:
            curDist = Levenshtein(t, w)
            if curDist < minCost:
                minCost = curDist
                target = t
    elif mode == 2:
        for t in dictionary:
            curDist = OptimalStringAlign(t, w)
            #curDist = DamerauLevenshtein(t, w)
            if curDist < minCost:
                minCost = curDist
                target = t
    else:
        target = DamerauLevenshteinTrie(w)

    return target

def DamerauLevenshteinTrie(w):
    minDist = [len(w)]
    target = ['none']
    curRow = range(len(w)+1)
    for letter in trie.children:
        searchRec(trie.children[letter], letter, None, w, curRow, None, target, minDist)
    return target[0]

def searchRec(node, letter, pletter, w, prevRow, pprevRow,target, minDist):
    cols = len(w) +1
    curRow = [prevRow[0]+1]
    if w == 'roxas' and node.word == 'rosas':
        a = 1
    for col in xrange(1, cols):
        insert = curRow[col-1]+1
        delete = prevRow[col]+1
        replace = prevRow[col-1] + (w[col-1] != letter)
        swap = sys.maxint
        if cols-1 > 0 and pprevRow != None and w[col-1] == pletter and w[cols-2] == letter:
            swap = pprevRow[col-2]+1
        else:
            swap = sys.maxint
        curRow.append(min(insert, delete, replace, swap))

    if node.word and minDist[0] > curRow[-1]:
        target[0] = node.word
        minDist[0] = curRow[-1]

    if min(curRow) < len(w):
        for nletter in node.children:
            searchRec(node.children[nletter], nletter, letter,w, curRow, prevRow, target, minDist)

def DamerauLevenshtein(t, w):
    cols = len(t)
    rows = len(w)
    d = {}
    maxdist = cols + rows
    d[(-1, -1)] = maxdist

    for i in xrange(0, cols+1):
        d[(i, -1)] = maxdist
        d[(i, 0)] = i
    for j in xrange(0, rows+1):
        d[(-1, j)] = maxdist
        d[(0, j)] = j

    charMap = {}
    for c in t:
        charMap[c] = 0
    for c in w:
        charMap[c] = 0

    # substring dists
    for i in xrange(1, cols+1):
        sub = 0
        for j in xrange(1, rows+1):
            i1 = charMap[w[j-1]]
            j1 = sub
            cost = 0
            if t[i-1] == w[j-1]:
                sub = j-1
            else:
                cost = 1
            delete = d[(i-1, j)]+1
            insert = d[(i, j-1)]+1
            replace = d[(i-1, j-1)] + cost
            d[(i, j)] = min(delete, insert, replace)
            if i1 > 0 and j1 > 0:
                d[(i, j)] = min(d[(i,j)], d[(i1-1,j1-1)]+(i-i1-1)+(j-j1-1)+1)
        charMap[t[i-1]] = i
    return d[(cols, rows)]

def OptimalStringAlign(t, w):
    cols = len(t)
    rows = len(w)
    d = {}
    for i in xrange(-1, cols+1):
        d[(i, -1)] = i+1
    for j in xrange(-1, rows+1):
        d[(-1, j)] = j+1

    for i in xrange(0, cols):
        for j in xrange(0, rows):
            cost = int(t[i] != w[j])
            delete = d[(i-1, j)]+1
            insert = d[(i, j-1)]+1
            replace = d[(i-1, j-1)] + cost
            d[(i, j)] = min(delete, insert, replace)
            if i > 0 and j > 0 and t[i] == w[j-1] and w[j] == t[i-1]:
                d[(i, j)] = min(d[(i, j)], d[(i-2, j-2)]+cost)
    return d[cols-1, rows-1]

# main file
if __name__ == "__main__":
    start = time.time()
    global dictionary, raw
    dictFile = open(sys.argv[3], 'r')
    dictionary = dictFile.read().splitlines()
    dictFile.close()
    rawFile = open(sys.argv[2], 'r')
    raw = rawFile.read().splitlines()
    rawFile.close()
    outFile = open(sys.argv[4], 'w')
    if sys.argv[1] not in ['1', '2', '3']:
        sys.stdout.write('invalid mode!')
        exit()
    mode = int(sys.argv[1])
    outList = []
    global trie
    trie = TrieNode()

    if mode == 3:
        for t in dictionary:
            trie.insert(t)

    for w in raw:
        outList.append(search(w, mode)+'\n')
    outFile.writelines(outList)
    outFile.close()
    print time.time()-start
    exit()

