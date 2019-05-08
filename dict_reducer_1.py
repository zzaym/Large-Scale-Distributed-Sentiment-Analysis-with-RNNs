#!/usr/bin/python
'''
Reads in a list of [word 1], counts number of occurrences for each word, and ouputs a list of [word count].
'''

import sys

previous = None
sum = 0

for line in sys.stdin:
    key, value = line.split( '\t' )
    
    if key != previous:
        if previous is not None:
            print(previous + '\t' +str(sum))
        previous = key
        sum = 0
    
    sum = sum + int( value )

print(previous + '\t' +str(sum))
