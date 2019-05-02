#!/usr/bin/python
import sys
from collections import Counter
import pickle
import boto3

res = []
scores = []
prev_text = None
for line in sys.stdin:
    try:
        text, score = line.split( '\t' )
        
        if text!=prev_text:
            if prev_text is not None:
                c = Counter(scores)
                top_scores = c.most_common(5)
                top_scores.sort()
                res.append((top_scores[0][0],prev_text))
            prev_text = text
            scores = []
        scores.append(score[:-1])
    except ValueError: pass

top_scores = c.most_common(5)
top_scores.sort()
res.append((top_scores[0][0],prev_text))
obj = pickle.dumps(res)

# save to s3 
bucket = 'YOUR BUCKET'
key = 'data.p'
s3 = boto3.resource('s3')
s3.Object(bucket, key).put(Body=obj)