#!/usr/bin/python

'''
This file is used to change words to their respective index according to a pre-loaded dictionary, 
truncate or pad each sentence to make it a pre-specified fixed length(100), 
remove duplicate and keep the mode of ratings and map ratings to binary sentiment indicator
'''

import sys
import json
import h5py
import numpy as np
from collections import Counter
import boto3
import socket

binary_dict = {1:0,2:0,3:0,4:1,5:1}

s3 = boto3.client('s3')
s3.download_file('cs205amazonreview','vocab_10000.json','vocab_10000.json')

with open('vocab_10000.json') as f:
    vocab_dict = json.load(f)
    f.close()

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

h5file = h5py.File("result.h5", "w")

review_id = 0
prev_text = None
scores = []

for line in sys.stdin:
    try:
        text, score = line.split( '\t' )
        
        if text!=prev_text:
            if prev_text is not None:
                top_scores = Counter(scores).most_common(5)
                top_scores.sort()
                output = tokenize(prev_text)
                output.append(binary_dict[int(top_scores[0][0])])
                output = np.array(output)
                dset = h5file.create_dataset(str(review_id),output.shape,output.dtype,data = output)
                review_id += 1
            prev_text = text
            scores = []
        scores.append(score[:-1])
    except ValueError: pass

if prev_text is not None:
    top_scores = Counter(scores).most_common(5)
    top_scores.sort()
    output = tokenize(prev_text)
    output.append(binary_dict[int(top_scores[0][0])])
    output = np.array(output)
    dset = h5file.create_dataset(str(review_id),output.shape,output.dtype,data = output)
    review_id += 1

h5file.close()

node_name = socket.gethostname()
s3.upload_file('result.h5','cs205amazonreview',node_name+'_result.h5')
