#!/usr/bin/python

import sys
import json
import h5py
import numpy as np
from collections import Counter
import boto3
import socket

s3 = boto3.client('s3')
s3.download_file('cs205amazonreview','vocab_10000.json','vocab_10000.json')

with open('vocab_10000.json') as f:
    vocab_dict = json.load(f)
    f.close()

binary_dict = {1:0,2:0,3:0,4:1,5:1}

def tokenize(text, text_size = 100):
    # 1: padding
    # 2: unknown
    # 3 - vocab_size+2: vacab
    result = []
    for word in text.split():
        try: result.append(vocab_dict[word])
        except KeyError: result.append(2)
    result = result[:text_size]
    for i in range(text_size - len(result)):
        result.append(1)
    return result

prev_text = None
scores = []
data = []

for line in sys.stdin:
    try:
        text, score = line.split( '\t' )
        
        if text!=prev_text:
            if prev_text is not None:
                top_scores = Counter(scores).most_common(5)
                top_scores.sort()
                output = tokenize(prev_text)
                output.append(binary_dict[int(top_scores[0][0])])
                data.append(output)
            prev_text = text
            scores = []
        scores.append(score[:-1])
    except ValueError: pass

if prev_text is not None:
    top_scores = Counter(scores).most_common(5)
    top_scores.sort()
    output = tokenize(prev_text)
    output.append(binary_dict[int(top_scores[0][0])])
    data.append(output)

data = np.array(data)
h5file = h5py.File("result.h5", "w")
dset = h5file.create_dataset('data',data.shape,data.dtype,data = data)
h5file.close()

node_name = socket.gethostname()
s3.upload_file('result.h5','cs205amazonreview',node_name+'_result.h5')
