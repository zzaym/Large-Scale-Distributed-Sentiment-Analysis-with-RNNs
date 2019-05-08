#!/usr/bin/python

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
