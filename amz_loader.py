from torch.utils.data import Dataset
from torch.autograd import Variable
import pickle

class DatasetAmazon(Dataset):
    def __init__(self, path):
    	with open(path, 'rb') as f:
        	self.data = pickle.load(f)

    def __len__(self):
    	return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index][:-1] # up to the last one is text
        label = self.data[index][-1]
        label = label - 1  
        return Variable(text), Variable(label)