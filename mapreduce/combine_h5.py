import boto3
import h5py
import numpy as np

'''
Execute this file to combine several h5py file into a single h5py file. Key is file name. 
Value corresponding to each key is a numpy array storing data chunk.
'''

bucket = 'bucket'
filenames = ['file1.h5', 'file2.h5']
combined_filename = 'combined.h5'

combined_file = h5py.File(combined_filename,'w')

s3 = boto3.client('s3')
for i,fn in enumerate(filenames):
    print(fn)
    s3.download_file(bucket,fn,fn)
    print('downloaded')
    single_file = h5py.File(fn,'r')
    data = []
    for key in single_file.keys():
        data.append(np.array(single_file[key]))
    single_file.close()
    data = np.array(data)
    dset = combined_file.create_dataset(str(i), shape=data.shape, dtype=data.dtype, data=data, chunks = (1,101))

combined_file.close()
