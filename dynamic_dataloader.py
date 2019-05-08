'''
DYNAMIC LOAD BALANCER

redistributes the data load and batch size for each GPU based on their given runtime.
'''
import math
import time
import numpy as np
import torch
from torch.utils import data
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

def undo_cumulative_sum(arr):
    res = arr.copy()
    res[1:] -= res[:-1]
    return res

def get_batch_data_split(perc_arr, total_batch, total_data):
    cum = perc_arr.cumsum()
    cum[-1] = 1.
    # find the split of batch for each GPU
    batch_split = np.round(cum*total_batch)
    batch_size_split = undo_cumulative_sum(batch_split).astype(int)
    # find the split of iterations for each GPU
    iter_split = np.round(cum*total_data)
    iter_split = undo_cumulative_sum(iter_split)
    # infer data split based on the restriction of same iterations
    iter_size = np.min(iter_split//batch_size_split)
    sampler_split = np.insert(iter_size*batch_split,0,0).astype(int)
    return batch_size_split, sampler_split

def get_dynamic_loader(loader, time_taken, total_batch):
    overhead = time.time()
    sampler = loader.sampler
    total_data = sampler.total_size
    local_rank = sampler.rank
    world_size = sampler.world_size
    time_list = [torch.zeros(1).cuda() for _ in range(world_size)]
    dist.all_gather(time_list, torch.tensor(time_taken).cuda())
    # normalize by the previous workload
    # to get the normalized time taken
    time_arr = torch.tensor(time_list).cpu().data.numpy()/sampler.perc_split
    inv = 1./time_arr
    perc_arr = inv/inv.sum()
    sampler.perc_split = perc_arr
    batch_size_split, data_split = get_batch_data_split(perc_arr, total_batch, total_data)
    print("new split", data_split)
    print("overhead time", time.time()-overhead)
    sampler.set_split(data_split)
    new_loader = data.DataLoader(loader.dataset,
        batch_size = int(batch_size_split[local_rank]),
        shuffle = False,
        sampler = sampler, num_workers = loader.num_workers)
    return new_loader

class DynamicDistributedSampler(DistributedSampler):

    def __init__(self, *args, **kwargs):
        super(DynamicDistributedSampler, self).__init__(*args, **kwargs)
        self.world_size = dist.get_world_size()
        self.split = None
        self.perc_split = np.ones(self.world_size)/self.world_size

    def __iter__(self):
        # deterministically shuffle based on epoch     
        if (self.split is None):
            return super(DynamicDistributedSampler, self).__iter__()
        else:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            print(self.split[self.rank], self.split[self.rank+1])
            indices = indices[self.split[self.rank]:self.split[self.rank+1]]
            return iter(indices)
        
    def set_split(self, split):
        self.split = split
