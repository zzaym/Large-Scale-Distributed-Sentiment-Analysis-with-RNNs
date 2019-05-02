import math
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class DynamicDistributedSampler(DistributedSampler):

    def __init__(self, **kwargs):
        super(DynamicDistributedSampler, self).__init__(**kwargs)
        self.split = None#[0, 15000, 60000]

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

    def update_load(self, epoch_time):
        time_list = [torch.zeros(1).cuda() for _ in range(self.world_size)]
        dist.all_gather(time_list, torch.tensor(epoch_time).cuda())
        print(time_list)
        time_arr = torch.tensor(time_list).cpu().data.numpy()
        inv = 1./time_arr
        cum = (inv/inv.sum()).cumsum()
        cum[-1] = 1.
        split = [0]+(cum*self.total_size).astype(int).tolist()
        print(split)