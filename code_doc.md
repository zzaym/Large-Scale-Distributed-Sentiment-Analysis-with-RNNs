### RNN_model.py
Class RNN:  classifier for sentiment analysis. Input is fixed length float vector representing a truncated sentence

### amz_loader.py
Class DatasetAmazon: customized dataset class inheriting from torch dataset module, reading multiple files stored in a single h5py dataset as input. __ getitem__ method will read the next file of input once the current file is exhausted.

### combine_h5.py: 
Execute this file to combine several h5py file into a single h5py file. Key is file name. Value corresponding to each key is a numpy array storing data chunk.

### dynamic_dataloader.py
undo_cumulative_sum:

get_batch_data_split

get_dynamic_loader

### dynamic_rnn.py
+ metric classes: Average, F1_score, and Accuracy:
+ train: perform 1 epoch of forward and backward action
+ get_dataloader:create dataset object, randomly split into 90% training data and 10% test data.
