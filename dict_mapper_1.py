#!/usr/bin/python

import sys

for line in sys.stdin:
    text = line.split(',')[1]
    for word in text.split():
        if word is not None:
            print(word+'\t1')
            
