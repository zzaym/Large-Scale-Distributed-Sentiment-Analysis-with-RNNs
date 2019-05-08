#!/usr/bin/python

'''
Reads in a list of [word count] and reorders to a list of [count word] to prepare for reverse sort.
'''

import sys

for line in sys.stdin:
    word,count = line.split('\t')
    print(count[:-1]+'\t'+word)

