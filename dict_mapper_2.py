#!/usr/bin/python

import sys

for line in sys.stdin:
    word,count = line.split('\t')
    print(count[:-1]+'\t'+word)

