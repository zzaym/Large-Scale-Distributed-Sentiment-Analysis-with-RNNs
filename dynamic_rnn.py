"""
Adapted from PyTorch 1.0 Distributed Trainer with Amazon AWS
"""

import time
import sys
import torch
import argparse
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from torch.multiprocessing import Pool, Process
from torch.utils.data import random_split
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
# from torch.utils.data.distributed import DistributedSampler
from dynamic_dataloader import DynamicDistributedSampler as DistributedSampler
from dynamic_dataloader import get_dynamic_loader
# from dynamic_dataparallel import DistributedDataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from amz_loader import DatasetAmazon
from sklearn.metrics import f1_score
from RNN_model import RNN

# the framework is inspired and adapted from https://pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html

class Average(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count

    def __str__(self):
        return '{:.6f}'.format(self.average)


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output, label):
        # accumulate counts necessary for acc calculation
        pred = output.ge(torch.zeros_like(output)).long().data
        correct = pred.eq(label.data).sum().item()

        self.correct += correct
        self.count += output.size(0)

    @property
    def accuracy(self):
        # perform calculation of accuracy with stored statistics
        return self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)


class F1_Score(object):
    ''' 
    accumulate statistics for F1 score for every batch trained, but only calculate F1 score when an epoch is finished
    '''
    def __init__(self):
        self.pos = torch.zeros(1)
        self.tp = torch.zeros(1)
        self.fn = torch.zeros(1)
        self.eps = 1e-9

    def update(self, output, label):
        pred = output.ge(torch.zeros_like(output)).float()
        label = label.float()
        self.pos += pred.sum()
        self.tp += (pred * label).sum()
        self.fn += ((1-pred) * (1-label)).sum()
        
    @property
    def f1_score(self):
        ''' 
        this function gives equal weights to both classes when calculating F1 score
        '''
        precision = self.tp.div(self.pos.add(self.eps))
        recall = self.tp.div(self.tp.add(self.fn).add(self.eps))
        return (precision*recall).div(precision+recall+self.eps).mul(2).item() 
    
    def __str__(self):
        return '{:.2f}'.format(self.f1_score)


class Trainer(object):
    def __init__(self, net, optimizer, train_loader, test_loader, loss, dynamic, filename):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss = loss
        self.timer = 0
        self.total_batch = train_loader.batch_size*dist.get_world_size()
        self.dynamic = dynamic
        self.filename = filename

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss = self.train()
            train_time = time.time() - epoch_start
            print("Train Time: ", train_time)
            test_loss, test_acc, test_f1 = self.evaluate()
            epoch_time = time.time()-epoch_start
            # updating the batch dynamically
            # self.train_loader.sampler.update_load(self.timer, 100)
            if (self.dynamic == 0 and epoch == 1) or self.dynamic > 0:
                self.train_loader = get_dynamic_loader(self.train_loader, self.timer, self.total_batch)
            print('Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {},'.format(train_loss),
                'test loss: {}, test acc: {}, test f1: {}.'.format(test_loss, test_acc, test_f1),
                'epoch time: {}'.format(epoch_time), flush = True)
            torch.save(self.net, self.filename)

    def train(self):
        train_loss = Average()
        # train_f1 = F1_Score()
        begin_time = time.time()

        print("Self.net.train: ", time.time()-begin_time)
        forward_timer = 0
        loss_timer = 0
        backward_timer = 0
        opti_timer = 0
        update_timer = 0
        total_timer = 0
        load_timer = 0
        count = 0
        load_start = time.time()

        i = 0
        self.net.train()
        for data, label in self.train_loader:
            load_timer += time.time()-load_start
            #start_time = time.time()
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            # forward is called here
            forward_start = time.time()
            output = self.net(data)
            forward_timer += time.time()-forward_start

            loss_start = time.time()
            loss = self.loss(output, label.float())
            
            self.optimizer.zero_grad()
            
            backward_start = time.time()
            loss_timer += backward_start - loss_start
            loss.backward()
            
            opti_start = time.time()
            backward_timer += opti_start - backward_start
            self.optimizer.step()
            
            update_start = time.time()
            opti_timer += update_start - opti_start
            train_loss.update(loss.item(), data.size(0))
            # train_f1.update(output, label)
            update_timer += time.time() - update_start
            # total_timer += time.time() - start_time
            load_start = time.time()

            i += 1
            if i % 2000 == 0:
                print('Iter {}, Train Loss: {}'.format(i, train_loss))
        self.timer = forward_timer
        print("Forward Time : {}s".format(forward_timer))
        print("Loss Time: ", loss_timer, "Backward Time: ", backward_timer, "Opti Time: ", opti_timer)
        print("Update Time", update_timer)
        print("Load Time", load_timer)
        print("------------------------------------------")
        return train_loss

    def evaluate(self):
        test_loss = Average()
        test_acc = Accuracy()
        test_f1 = F1_Score()

        self.net.eval()
        with torch.no_grad():
            for data, label in self.test_loader:
                data = data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                output = self.net(data)
                loss = self.loss(output, label.float())

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, label)
                test_f1.update(output, label)

        return test_loss, test_acc, test_f1


def get_dataloader(root, batch_size, workers = 0):
    amazon = DatasetAmazon(root)
    train_length = int(0.9 * len(amazon))
    test_length = len(amazon) - train_length
    amz_train, amz_test = random_split(amazon, (train_length,test_length))
    sampler = DistributedSampler(amz_train)
    train_loader = data.DataLoader(amz_train, shuffle=(sampler is None), batch_size=batch_size, \
                        sampler=sampler, num_workers=workers, drop_last=True)
    test_loader = data.DataLoader(amz_test, shuffle=False, batch_size=batch_size, num_workers=workers, drop_last=True)

    return train_loader, test_loader


if __name__ == '__main__':
    
    initial_time = time.time()
    print("Collect Inputs...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--dir", type=str, default='data.h5')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--n_vocab", type=int, default=1e4)
    parser.add_argument("--dynamic", type=int, default=-1)
    parser.add_argument("--filename", type=str, default='checkpoint.pth.tar')
    args = parser.parse_args()
    
    # number of vocabulary
    num_vocab = args.n_vocab

    # location of data file
    path = args.dir
   
    # Starting Learning Rate
    lr = args.lr

    # Batch Size for training and testing
    batch_size = args.batch
    
    # Number of additional worker processes for dataloading
    workers = args.workers

    # Number of epochs to train for
    num_epochs = args.epochs

    # dynmaic step
    # 0 only updates the dataloader dynamically after the 1st epoch
    # if not updates every given argument
    dynamic_step = args.dynamic

    # filename
    filename = args.filename

    # Distributed backend type
    dist_backend = 'nccl'
    print("Data Directory: {}".format(args.dir))
    print("Batch Size: {}".format(batch_size))
    print("Learning rate: {}".format(lr))
    print("Max Number of Epochs: {}".format(num_epochs))
    print("Initialize Process Group...")

    torch.cuda.set_device(args.local_rank)

    torch.distributed.init_process_group(backend=dist_backend,
                                         init_method='env://')
    torch.multiprocessing.set_start_method('spawn', force=True)


    # Establish Local Rank and set device on this node
    local_rank = args.local_rank
    dp_device_ids = [local_rank]

    print("Initialize Model...")
    # Construct Model
    model = RNN(num_vocab).cuda()

    # Make model DistributedDataParallel
    model = DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)

    # define loss function (criterion) and optimizer
    weight = torch.FloatTensor([0.2]).cuda()
    loss = nn.BCEWithLogitsLoss(pos_weight=weight).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print("Initialize Dataloaders...")
    train_loader, test_loader = get_dataloader(path, batch_size, workers)
    print("Training...")
    trainer = Trainer(model, optimizer, train_loader, test_loader, loss, dynamic_step, filename)
    trainer.fit(num_epochs)

    print("Total time: {:.3f}s".format(time.time()-initial_time))
