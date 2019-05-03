#!/usr/bin/python

import sys
import json
import h5py
import numpy as np
from collections import Counter

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

h5file = h5py.File("result_14000.h5", "w")
for word in vocab_dict.keys():
    num = np.array(vocab_dict[word])
    dset = h5file.create_dataset(word, num.shape, dtype=num.dtype, data = num)
    #dset[word] = num


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
                output.append(int(top_scores[0][0]))
                output = np.array(output)
                dset = h5file.create_dataset(str(review_id),output.shape,output.dtype,data = output)
                review_id += 1
            prev_text = text
            scores = []
        scores.append(score[:-1])
    except ValueError: pass

top_scores = Counter(scores).most_common(5)
top_scores.sort()
output = tokenize(prev_text)
output.append(int(top_scores[0][0]))
output = np.array(output)
dset = h5file.create_dataset(str(review_id),output.shape,output.dtype,data = output)
review_id += 1

h5file.close()