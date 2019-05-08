#!/bin/bash

'''
This file is used to perform bootstrap action in AWS EMR and install software.
'''
sudo easy_install pip
sudo /usr/local/bin/pip install --upgrade pip
sudo /usr/local/bin/pip install boto3
sudo /usr/local/bin/pip install h5py
