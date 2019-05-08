from torch.utils.data import Dataset
import torch
import h5pickle as h5py

'''
Class DatasetAmazon: customized dataset class inheriting from torch dataset module, which works with PyTorch distributed computing modules.
The __init__ function reads multiple files stored in a single h5py dataset as input. 
__ getitem__ method will read the next file of input once the current file is exhausted.
'''

class DatasetAmazon(Dataset):
    def __init__(self, path):
        self.f = h5py.File(path,'r')
        self.keyname = list(self.f.keys())
        self.size = min([len(self.f[key]) for key in self.keyname])
        
    def __len__(self):
        return len(self.keyname) * self.size - 1
    
    def __getitem__(self, index):
        line = self.f[self.keyname[int(index/self.size)]][index%self.size]
        text = line[:-1] # up to the last one is text
        label = line[-1:]
        label = (label > 3) * 1
        return torch.LongTensor(text), torch.LongTensor(label)
