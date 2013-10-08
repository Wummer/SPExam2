# + * len() range() sys.argv open() read()/readlines() split
from __future__ import division
import sys

lines=[l.strip().split() for l in open(sys.argv[1]).readlines()]
word_tokens=sum([len(l) for l in lines])

words={}
for l in lines:
    for w in l:
        if w not in words:
            words[w]=0
        words[w]+=1

for w in words:
    print w+'\t'+str(words[w]/word_tokens)
