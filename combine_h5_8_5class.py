import boto3
import h5py
import numpy as np

IPs = ['ip-172-31-73-255','ip-172-31-71-225','ip-172-31-79-251','ip-172-31-64-191',
       'ip-172-31-70-26','ip-172-31-70-228','ip-172-31-66-18','ip-172-31-78-81']

combined_file = h5py.File('combined_result.h5','w')

#s3 = boto3.client('s3')
for i,ip in enumerate(IPs):
    print(ip)
    #s3.download_file('cs205amazonreview',ip+'_result.h5',ip+'_result.h5')
    #print('downloaded')
    single_file = h5py.File(ip+'_result.h5','r')
    data = []
    for key in single_file.keys():
        data.append(np.array(single_file[key]))
    single_file.close()
    data = np.array(data)
    dset = combined_file.create_dataset(str(i), shape=data.shape, dtype=data.dtype, data=data, chunks = (1,101))

combined_file.close()