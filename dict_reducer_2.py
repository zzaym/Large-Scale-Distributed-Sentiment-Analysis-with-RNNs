#!/usr/bin/python

import sys

vocab_size = 10000

for i,line in enumerate(sys.stdin):
    if i >= vocab_size: break
    word = line.split('\t')[1][:-1]
    print('{\"'+word+'\":'+str(i+1)+'}')

