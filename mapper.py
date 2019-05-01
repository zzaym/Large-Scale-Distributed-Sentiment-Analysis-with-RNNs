#!/usr/bin/python

import sys
import json

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stop_words.update(['get','would'])

def process_text(text):
    content = re.sub("[^a-zA-Z' ]+",'',text).lower().split()
    x = [word for word in content if word not in stop_words]
    
    # text = re.sub( r'[^\w]', '', text ).lower().strip()
    # text = text.split()
    # purged_word_list = [word for word in text if word not in stop_words]
    return ' '.join(x)

for i,line in enumerate(sys.stdin):
    #if i == 14000: break
    data = json.loads(line.strip())
    #print(data)
    text = process_text(data['reviewText'])
    #print(text)
    if text != '':
        print(text+'\t'+str(int(data['overall'])))#+'\t'+data['reviewerID']+'\t'+data['asin'])
    