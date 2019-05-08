#!/usr/bin/python

'''
Reads in a reversely sorted list of [count word] and generates dictionary for the 10000 most frequent words.
'''

import sys

vocab_size = 10000

for i,line in enumerate(sys.stdin):
    if i >= vocab_size: break
    word = line.split('\t')[1][:-1]
    print('{\"'+word+'\":'+str(i+1)+'}')

